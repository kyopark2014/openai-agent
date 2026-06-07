"""OpenAI Agents SDK runtime for the Streamlit app (Bedrock + MCP + Skills)."""

from __future__ import annotations

import chat
import io
import json
import logging
import mcp_config
import os
import re
import skill
import subprocess
import sys
import sysconfig
import traceback
import utils

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import parse

import boto3
from agents import (
    Agent,
    RunConfig,
    Runner,
    SQLiteSession,
    SessionSettings,
    function_tool,
    set_default_openai_client,
)
from agents.exceptions import ModelBehaviorError
from agents.mcp import MCPServerManager, MCPServerStdio, MCPServerStreamableHttp
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from aws_bedrock_token_generator import provide_token
from openai import AsyncBedrockOpenAI, BadRequestError

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("openai-agent")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")
CONVERSATIONS_DB = os.path.join(WORKING_DIR, ".conversations.db")
CONVERSATION_HISTORY_LIMIT = 24
BASE_SYSTEM_PROMPT = skill.AGENT_BASE_PROMPT

config = utils.load_config()
sharing_url = config.get("sharing_url")

_RUN_CONFIG = RunConfig(session_settings=SessionSettings(limit=CONVERSATION_HISTORY_LIMIT))

_agent: Agent | None = None
_selected_tools: list[str] = []
_selected_mcp: list[str] = []
_selected_skills: list[str] = []
_last_region: str | None = None
_last_model: str | None = None

_ARTIFACT_EXT = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"}
)
_mpl_runtime_ready = False

_exec_globals: dict[str, Any] = {
    "__builtins__": __builtins__,
    "subprocess": subprocess,
    "json": json,
    "os": os,
    "sys": sys,
    "io": io,
    "pathlib": Path,
    "WORKING_DIR": WORKING_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
}


def _s3_console_url(uri: str, region: str) -> str:
    if not uri or not uri.startswith("s3://"):
        return ""
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return f"https://{region}.console.aws.amazon.com/s3/object/{bucket}?prefix={parse.quote(key, safe='')}"


def _artifact_mtime_snapshot() -> dict[str, float]:
    snap: dict[str, float] = {}
    if not os.path.isdir(ARTIFACTS_DIR):
        return snap
    for dirpath, _, filenames in os.walk(ARTIFACTS_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                snap[os.path.relpath(full, WORKING_DIR)] = os.path.getmtime(full)
            except OSError:
                pass
    return snap


def _touched_paths(before: dict[str, float], after: dict[str, float]) -> list[str]:
    return sorted(rel for rel, mt in after.items() if rel not in before or before[rel] != mt)


def _paths_for_ui(relative_paths: list[str]) -> list[str]:
    return [os.path.abspath(os.path.join(WORKING_DIR, rel)) for rel in relative_paths]


def _ensure_matplotlib_runtime() -> None:
    global _mpl_runtime_ready
    if _mpl_runtime_ready:
        return
    try:
        import warnings

        import matplotlib
        import matplotlib as mpl

        matplotlib.use("Agg")
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=r"FigureCanvasAgg is non-interactive.*", category=UserWarning
        )
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [
            "AppleGothic",
            "Apple SD Gothic Neo",
            "Malgun Gothic",
            "NanumGothic",
            "Noto Sans CJK KR",
            "DejaVu Sans",
            "sans-serif",
        ]
    except Exception as exc:
        logger.info("matplotlib runtime setup skipped: %s", exc)
    _mpl_runtime_ready = True


def _ensure_cli_scripts_on_path() -> None:
    import site

    extra: list[str] = []
    user_base = getattr(site, "USER_BASE", None)
    if user_base:
        user_bin = os.path.join(user_base, "bin")
        if os.path.isdir(user_bin):
            extra.append(user_bin)
    try:
        scripts = sysconfig.get_path("scripts")
        if scripts and os.path.isdir(scripts):
            extra.append(scripts)
    except Exception:
        pass
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for d in reversed(extra):
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


@function_tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    before_files = _artifact_mtime_snapshot()
    old_cwd = os.getcwd()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        os.chdir(WORKING_DIR)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        _ensure_matplotlib_runtime()
        exec(code, _exec_globals)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        result = output or ""
        if errors:
            result += f"\n[stderr]\n{errors}"
        if not result.strip():
            result = "Code executed successfully (no output)."

        touched = _touched_paths(before_files, _artifact_mtime_snapshot())
        artifact_rels = [r for r in touched if os.path.splitext(r)[1].lower() in _ARTIFACT_EXT]
        other_rels = [r for r in touched if r not in artifact_rels]
        if other_rels:
            lines = "\n".join(os.path.abspath(os.path.join(WORKING_DIR, r)) for r in other_rels)
            result += f"\n[artifacts]\n{lines}"
        if artifact_rels:
            return json.dumps({"output": result.strip(), "path": _paths_for_ui(artifact_rels)}, ensure_ascii=False)
        return result
    except Exception:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        tb = traceback.format_exc()
        logger.error("Code execution error: %s", tb)
        return f"Error executing code:\n{tb}"


@function_tool
def upload_file_to_s3(filepath: str) -> str:
    """Upload a local file to S3 and return the download URL."""
    bucket = config.get("s3_bucket")
    if not bucket:
        return "S3 bucket is not configured."
    full_path = os.path.join(WORKING_DIR, filepath)
    if not os.path.exists(full_path):
        return f"File not found: {filepath}"
    try:
        region = config.get("region", "us-west-2")
        content_type = utils.get_contents_type(filepath)
        client = boto3.client("s3", region_name=region)
        with open(full_path, "rb") as f:
            client.put_object(Bucket=bucket, Key=filepath, Body=f.read(), ContentType=content_type)
        if sharing_url:
            return f"Upload complete: {sharing_url}/{parse.quote(filepath)}"
        return f"Upload complete: {_s3_console_url(f's3://{bucket}/{filepath}', region)}"
    except Exception as e:
        return f"Upload failed: {e}"


@function_tool
def load_skill(skill_name: str) -> str:
    """Load a skill into .agents/<skill_name>/ for file_read or bash."""
    result = skill.load_skill(skill_name)
    return json.dumps(result, ensure_ascii=False)


@function_tool
def bash(command: str) -> str:
    """Execute a bash command and return the result."""
    _ensure_cli_scripts_on_path()
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=WORKING_DIR,
        timeout=300,
        env=os.environ,
    )
    parts = []
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.returncode != 0:
        parts.append(f"Return code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"


@function_tool
def current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@function_tool
def file_read(path: str, from_line: int = 0, lines: int = 0) -> str:
    """Read a text file under the application working directory."""
    full_path = Path(WORKING_DIR) / path
    if not full_path.is_file():
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    try:
        content = full_path.read_text(encoding="utf-8")
        if from_line > 0 or lines > 0:
            all_lines = content.split("\n")
            start = max(0, from_line - 1)
            end = start + lines if lines > 0 else len(all_lines)
            content = "\n".join(all_lines[start:end])
        return json.dumps({"path": path, "text": content}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "path": path}, ensure_ascii=False)


@function_tool
def file_write(path: str, content: str) -> str:
    """Write text to a file. Generated outputs are saved under artifacts/."""
    rel_path = _resolve_write_path(path)
    full_path = Path(WORKING_DIR) / rel_path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return json.dumps({"path": rel_path, "status": "ok"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "path": rel_path}, ensure_ascii=False)


OPTIONAL_TOOLS = {
    "current_time": current_time,
    "file_read": file_read,
    "file_write": file_write,
}

# Paths that file_write may target outside artifacts/ (skill workspace, etc.)
_WRITE_ALLOWED_PREFIXES = (".agents/", "skills/", "memory/", "artifacts/")


def _resolve_write_path(path: str) -> str:
    """Route generated files into artifacts/ unless writing to an allowed workspace path."""
    normalized = path.replace("\\", "/").lstrip("/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if any(normalized.startswith(prefix) for prefix in _WRITE_ALLOWED_PREFIXES):
        return normalized
    return f"artifacts/{normalized}"


def _configure_bedrock() -> AsyncBedrockOpenAI:
    global _last_region, _last_model
    region = chat.get_responses_region()
    model = chat.get_responses_model_id()
    client = AsyncBedrockOpenAI(
        aws_region=region,
        bedrock_token_provider=lambda r=region: provide_token(region=r),
    )
    set_default_openai_client(client)
    _last_region = region
    _last_model = model
    logger.info("Bedrock client configured model=%s region=%s", model, region)
    return client


def _bedrock_stale() -> bool:
    return _last_region != chat.get_responses_region() or _last_model != chat.get_responses_model_id()


def session_for(user_key: str | None = None) -> SQLiteSession:
    """SQLiteSession — Runner loads/saves history automatically."""
    return SQLiteSession(user_key or chat.user_id or "default", CONVERSATIONS_DB)


async def clear_agent_session(user_key: str | None = None) -> None:
    session = session_for(user_key)
    await session.clear_session()
    session.close()


_conversations_reset_for_process = False


def reset_conversations_db_at_startup() -> None:
    """Delete the SQLite session DB once per Streamlit server process."""
    global _conversations_reset_for_process
    if _conversations_reset_for_process:
        return
    _conversations_reset_for_process = True
    for suffix in ("", "-wal", "-shm"):
        path = CONVERSATIONS_DB + suffix
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    logger.info("Conversations database reset at app startup")


def _build_mcp_servers(server_names: list[str]) -> list[Any]:
    servers: list[Any] = []
    for tool_name in server_names:
        mcp = mcp_config.load_config(tool_name)
        if not mcp or "mcpServers" not in mcp:
            logger.warning("No MCP config for: %s", tool_name)
            continue

        server_key = next(iter(mcp["mcpServers"]))
        server_config = mcp["mcpServers"][server_key]

        if server_config.get("type") == "streamable_http":
            url = server_config["url"]
            headers = server_config.get("headers", {})
            logger.info("MCP streamable_http: %s -> %s", tool_name, url)
            servers.append(
                MCPServerStreamableHttp(
                    params={"url": url, "headers": headers},
                    name=tool_name,
                    cache_tools_list=True,
                )
            )
            continue

        command = server_config["command"]
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        cmd_path = os.path.expanduser(command) if isinstance(command, str) else str(command)
        if ("/" in cmd_path or (isinstance(command, str) and command.startswith("~"))) and not os.path.isfile(
            cmd_path
        ):
            logger.warning("Skipping MCP %s: executable not found at %s", tool_name, cmd_path)
            continue

        logger.info("MCP stdio: %s command=%s args=%s", tool_name, command, args)
        servers.append(
            MCPServerStdio(
                params={"command": command, "args": args, "env": env},
                name=tool_name,
                cache_tools_list=True,
            )
        )
    return servers


def _build_tools(optional_tool_names: list[str], *, skills_enabled: bool) -> list[Any]:
    tools: list[Any] = [execute_code, bash, upload_file_to_s3]
    if skills_enabled:
        tools.append(load_skill)
    for name in optional_tool_names:
        fn = OPTIONAL_TOOLS.get(name)
        if fn:
            tools.append(fn)
    return tools


def create_agent(
    optional_tool_names: list[str],
    mcp_server_names: list[str],
    skill_list: list[str],
) -> Agent:
    _configure_bedrock()
    skills_enabled = chat.skill_mode == "Enable"
    tools = _build_tools(optional_tool_names, skills_enabled=skills_enabled)
    mcp_servers = _build_mcp_servers(mcp_server_names)
    model = chat.get_responses_model_id()
    instructions = (
        skill.build_agent_instructions(skill_list)
        if skills_enabled
        else BASE_SYSTEM_PROMPT
    )
    if skills_enabled:
        logger.info("selected skills: %s", skill_list)
    logger.info("Creating Agent model=%s tools=%d mcp=%d", model, len(tools), len(mcp_servers))
    return Agent(
        name="서연",
        instructions=instructions,
        model=model,
        tools=tools,
        mcp_servers=mcp_servers,
    )


def _text_from_run_item(event: RunItemStreamEvent) -> str | None:
    item = event.item
    raw = getattr(item, "raw_item", None)
    if raw is None or not getattr(raw, "content", None):
        return None
    for block in raw.content:
        if getattr(block, "text", None):
            return block.text
    return None


async def agent_stream(
    agent: Agent,
    query: str,
    queue: Any,
    session: SQLiteSession,
) -> tuple[str, list[str], list[dict], str]:
    image_urls: list[str] = []
    references: list[dict] = []
    current = ""
    final_result = ""

    result = Runner.run_streamed(
        agent,
        query,
        max_turns=20,
        session=session,
        run_config=_RUN_CONFIG,
    )

    try:
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if getattr(data, "type", None) == "response.output_text.delta":
                    delta = getattr(data, "delta", "") or ""
                    if delta:
                        current += delta
                        if queue is not None:
                            queue.stream(current)
                continue

            if not isinstance(event, RunItemStreamEvent):
                continue

            if event.name == "tool_called":
                raw = getattr(event.item, "raw_item", None)
                if raw is not None and queue is not None:
                    tool_name = getattr(raw, "name", "") or ""
                    tool_use_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or tool_name
                    args = getattr(raw, "arguments", "")
                    queue.register_tool(tool_use_id, tool_name)
                    queue.tool_update(tool_use_id, f"Tool: {tool_name}, Input: {args}")
                    current = ""

            elif event.name == "tool_output":
                raw = getattr(event.item, "raw_item", None)
                if raw is not None:
                    if isinstance(raw, dict):
                        output = raw.get("output", "")
                        tool_use_id = raw.get("call_id", "")
                    else:
                        output = getattr(raw, "output", "") or ""
                        tool_use_id = getattr(raw, "call_id", "") or ""
                    tool_name = queue.get_tool_name(tool_use_id) if queue is not None else ""
                    if queue is not None:
                        queue.notify(f"Tool Result ({tool_name}): {str(output)[:500]}")
                    _info, urls, refs = chat.get_tool_info(tool_name, str(output))
                    references.extend(refs or [])
                    image_urls.extend(urls or [])

            elif event.name == "message_output_created":
                text = _text_from_run_item(event)
                if text:
                    final_result = text

    except ModelBehaviorError as exc:
        logger.error("Agent model error: %s", exc)
        if queue is not None:
            queue.notify(f"Model error: {exc}")
        if not final_result and not current:
            final_result = (
                "Bedrock API 처리 중 서버 오류가 발생했습니다. "
                "대화가 길거나 도구 결과가 많으면 **대화 초기화** 후 다시 시도해 주세요."
            )
    except BadRequestError as exc:
        logger.error("Agent API validation error: %s", exc)
        if queue is not None:
            queue.notify(f"API validation error: {exc}")
        if not final_result and not current:
            final_result = (
                "대화 기록에 도구 호출 정보가 맞지 않아 요청이 거부되었습니다. "
                "**대화 초기화** 후 다시 시도해 주세요."
            )
    except Exception as exc:
        logger.error("Agent stream error: %s", exc)
        if queue is not None:
            queue.notify(f"Agent error: {exc}")
        if not final_result and not current:
            raise

    if not final_result and current:
        final_result = current
    return final_result, image_urls, references, current


async def run_agent(
    query: str,
    optional_tool_names: list[str],
    mcp_server_names: list[str],
    skill_list: list[str],
    notification_queue: Any,
) -> tuple[str, list[str]]:
    global _agent, _selected_tools, _selected_mcp, _selected_skills

    if (
        _selected_tools != optional_tool_names
        or _selected_mcp != mcp_server_names
        or _selected_skills != skill_list
        or _agent is None
        or _bedrock_stale()
    ):
        _selected_tools = list(optional_tool_names)
        _selected_mcp = list(mcp_server_names)
        _selected_skills = list(skill_list)
        _agent = create_agent(optional_tool_names, mcp_server_names, skill_list)
        logger.info("Agent recreated (config changed)")

    session = session_for()
    queue = notification_queue
    if queue is not None:
        queue.reset()

    mcp_servers = list(_agent.mcp_servers)
    if mcp_servers:
        async with MCPServerManager(mcp_servers, drop_failed_servers=True, strict=False) as manager:
            _agent.mcp_servers = manager.active_servers
            for server, err in manager.errors.items():
                logger.warning("MCP server '%s' failed: %s", getattr(server, "name", server), err)
            if not manager.active_servers:
                raise RuntimeError("No MCP servers connected. Check MCP configuration and dependencies.")
            final_result, image_urls, references, _ = await agent_stream(_agent, query, queue, session)
    else:
        final_result, image_urls, references, _ = await agent_stream(_agent, query, queue, session)

    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            content = reference["content"][:100].replace("\n", "")
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {content}...\n"
        final_result += ref

    if queue is not None:
        queue.result(final_result)
    return final_result, image_urls
