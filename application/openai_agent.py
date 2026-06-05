import chat
import json
import os
import logging
import sys
import utils
import boto3
import yaml
import skill
import subprocess
import mcp_config

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from agents import Agent, Runner, function_tool, set_default_openai_client
from agents.exceptions import ModelBehaviorError
from agents.mcp import MCPServerManager, MCPServerStdio, MCPServerStreamableHttp
from agents.memory.session_settings import SessionSettings
from conversations_session import ConversationsSession
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from aws_bedrock_token_generator import provide_token
from openai import AsyncBedrockOpenAI
from urllib import parse

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("openai-agent")

agent_tool_names = []
mcp_servers = []

tool_list = []

s3_prefix = "docs"
capture_prefix = "captures"

config = utils.load_config()
s3_bucket = config.get("s3_bucket")
sharing_url = config.get("sharing_url")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(WORKING_DIR, "skills")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")
CONVERSATIONS_DB = os.path.join(WORKING_DIR, ".conversations.db")
CONVERSATIONS_DIR = os.path.join(WORKING_DIR, ".conversations")
CONVERSATION_HISTORY_LIMIT = 12


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: str

class SkillManager:
    """Discovers, loads and selects Agent Skills following the Anthropic spec."""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.registry: dict[str, Skill] = {}
        self._discover()

    # ---- discovery & metadata loading ----

    def _discover(self):
        """Scan skills directory and load metadata (frontmatter only)."""
        if not os.path.isdir(self.skills_dir):
            os.makedirs(self.skills_dir, exist_ok=True)
            logger.info(f"Created skills directory: {self.skills_dir}")
            return

        for entry in os.listdir(self.skills_dir):
            skill_md = os.path.join(self.skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_md):
                try:
                    meta, instructions = self._parse_skill_md(skill_md)
                    skill = Skill(
                        name=meta.get("name", entry),
                        description=meta.get("description", ""),
                        instructions=instructions,
                        path=os.path.join(self.skills_dir, entry),
                    )
                    self.registry[skill.name] = skill
                    logger.info(f"Skill discovered: {skill.name}")
                except Exception as e:
                    logger.warning(f"Failed to load skill '{entry}': {e}")

    @staticmethod
    def _parse_skill_md(filepath: str) -> tuple[dict, str]:
        """Parse YAML frontmatter + markdown body from a SKILL.md file."""
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        if not raw.startswith("---"):
            return {}, raw

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw

        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return frontmatter, body

    # ---- prompt generation (progressive disclosure) ----
    def available_skills_xml(self) -> str:
        """Generate <available_skills> XML for the system prompt (metadata only)."""
        if not self.registry:
            return ""
        lines = ["<available_skills>"]
        for s in self.registry.values():
            lines.append("  <skill>")
            lines.append(f"    <name>{s.name}</name>")
            lines.append(f"    <description>{s.description}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_skill_instructions(self, name: str) -> Optional[str]:
        """Return full instructions for a skill (loaded on demand)."""
        skill = self.registry.get(name)
        return skill.instructions if skill else None

    def select_skills(self, query: str) -> list[Skill]:
        """Keyword-based matching to select relevant skills for a query."""
        query_lower = query.lower()
        selected = []
        for skill in self.registry.values():
            keywords = skill.description.lower().split()
            if any(kw in query_lower for kw in keywords if len(kw) > 3):
                selected.append(skill)
        return selected

    def build_active_skill_prompt(self, skills: list[Skill]) -> str:
        """Build the full instructions block for activated skills."""
        if not skills:
            return ""
        parts = ["<active_skills>"]
        for s in skills:
            parts.append(f'<skill name="{s.name}">')
            parts.append(s.instructions)
            parts.append("</skill>")
        parts.append("</active_skills>")
        return "\n".join(parts)

# global singleton
skill_manager = SkillManager()

SKILL_USAGE_GUIDE = (
    "\n## Skill 사용 가이드\n"
    "위의 <available_skills>에 나열된 skill이 사용자의 요청과 관련될 때:\n"
    "1. 먼저 get_skill_instructions 도구로 해당 skill의 상세 지침을 로드하세요.\n"
    "2. 지침에 포함된 코드 패턴을 execute_code 도구로 실행하세요.\n"
    "3. skill 지침이 없는 일반 질문은 직접 답변하세요.\n"
)

BASE_SYSTEM_PROMPT = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다.\n"
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다.\n"
    "모르는 질문을 받으면 솔직히 모른다고 말합니다.\n"
    "한국어로 답변하세요.\n\n"
    "## Agent Workflow\n"
    "1. 사용자 입력을 받는다\n"
    "2. 요청에 맞는 skill이 있으면 get_skill_instructions 도구로 상세 지침을 로드한다\n"
    "3. skill 지침에 따라 execute_code, write_file 등의 도구를 사용하여 작업을 수행한다\n"
    "4. 결과 파일이 있으면 upload_file_to_s3로 업로드하여 URL을 제공한다\n"
    "5. 최종 결과를 사용자에게 전달한다\n"
)

def build_system_prompt(plugin_name: Optional[str] = None, command: Optional[str] = None) -> str:
    """Assemble the full system prompt with available skills metadata."""
    if command:
        base = skill.build_command_prompt(plugin_name, command)
    elif plugin_name:
        base = skill.build_skill_prompt(plugin_name)
    else:
        base = BASE_SYSTEM_PROMPT

    return base

def s3_uri_to_console_url(uri: str, region: str) -> str:
    """Open the object in the AWS S3 console (when sharing_url is not configured)."""
    if not uri or not uri.startswith("s3://"):
        return ""
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    enc_key = parse.quote(key, safe="")
    return f"https://{region}.console.aws.amazon.com/s3/object/{bucket}?prefix={enc_key}"

import io, os, sys, json, traceback
import subprocess as _subprocess, pathlib as _pathlib, shutil as _shutil
import tempfile as _tempfile, glob as _glob, datetime as _datetime
import math as _math, re as _re, requests as _requests
from pathlib import Path

_ARTIFACT_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"})

_mpl_runtime_ready = False

def _artifact_files_mtime_snapshot() -> dict:
    """Relative path from WORKING_DIR -> mtime. Only scans under artifacts/."""
    snap = {}
    if not os.path.isdir(ARTIFACTS_DIR):
        return snap
    for dirpath, _, filenames in os.walk(ARTIFACTS_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                rel = os.path.relpath(full, WORKING_DIR)
                snap[rel] = os.path.getmtime(full)
            except OSError:
                pass
    return snap


def _touched_artifact_paths(before: dict, after: dict) -> list:
    """Only files created or modified between pre/post execution snapshots."""
    touched = []
    for rel, mt in after.items():
        if rel not in before or before[rel] != mt:
            touched.append(rel)
    return sorted(touched)


def _paths_for_ui(relative_paths: list) -> list:
    """absolute path for Streamlit st.image."""
    out = []
    for rel in relative_paths:
            out.append(os.path.abspath(os.path.join(WORKING_DIR, rel)))
    return out


def _ensure_matplotlib_runtime():
    """Use non-interactive Agg backend, prefer CJK-capable fonts, silence headless/show noise."""
    global _mpl_runtime_ready
    if _mpl_runtime_ready:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")

        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"FigureCanvasAgg is non-interactive.*",
            category=UserWarning,
        )

        import matplotlib.font_manager as fm
        import matplotlib as mpl

        mpl.rcParams["axes.unicode_minus"] = False
        cjk_candidates = (
            "AppleGothic",
            "Apple SD Gothic Neo",
            "Malgun Gothic",
            "NanumGothic",
            "NanumBarunGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        )
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = list(cjk_candidates) + ["DejaVu Sans", "sans-serif"]

        _mpl_runtime_ready = True
    except Exception as e:
        logger.info(f"matplotlib runtime setup skipped: {e}")
        _mpl_runtime_ready = True

_exec_globals = {
    "__builtins__": __builtins__,
    "subprocess": _subprocess,
    "json": json,
    "os": os,
    "sys": sys,
    "io": io,
    "pathlib": _pathlib,
    "shutil": _shutil,
    "tempfile": _tempfile,
    "glob": _glob,
    "datetime": _datetime,
    "math": _math,
    "re": _re,
    "requests": _requests,
    "WORKING_DIR": WORKING_DIR,
    "ARTIFACTS_DIR": ARTIFACTS_DIR,
}

@function_tool
def execute_code(code: str) -> str:
    """Execute Python code and return stdout/stderr output.

    Use this tool to run Python code for tasks such as processing data,
    processing data, or performing computations. The execution environment
    has access to common libraries: pandas, numpy, matplotlib, seaborn, etc.
    json, csv, os, requests, etc.

    Variables and imports from previous calls persist across invocations.
    Generated files should be saved to the 'artifacts/' directory.

    Path variables (pre-defined, do NOT redefine):
    - WORKING_DIR: absolute path to application directory
    - ARTIFACTS_DIR: absolute path to artifacts directory (WORKING_DIR/artifacts)

    Args:
        code: Python code to execute.

    Returns:
        Captured stdout output, or error traceback if execution failed.
        If there is a result file, return the path of the file.            
    """
    logger.info(f"###### execute_code ######")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    before_files = _artifact_files_mtime_snapshot()

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

        result = ""
        if output:
            result += output
        if errors:
            result += f"\n[stderr]\n{errors}"
        if not result.strip():
            result = "Code executed successfully (no output)."

        after_files = _artifact_files_mtime_snapshot()
        touched = _touched_artifact_paths(before_files, after_files)
        artifact_rels = [
            r
            for r in touched
            if os.path.splitext(r)[1].lower() in _ARTIFACT_EXT
        ]
        other_rels = [r for r in touched if r not in artifact_rels]
        if other_rels:
            lines = "\n".join(
                os.path.abspath(os.path.join(WORKING_DIR, r)) for r in other_rels
            )
            result += f"\n[artifacts]\n{lines}"

        if artifact_rels:
            payload = {"output": result.strip()}
            payload["path"] = _paths_for_ui(artifact_rels)
            return json.dumps(payload, ensure_ascii=False)

        return result

    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.chdir(old_cwd)
        tb = traceback.format_exc()
        logger.error(f"Code execution error: {tb}")
        return f"Error executing code:\n{tb}"


@function_tool
def upload_file_to_s3(filepath: str) -> str:
    """Upload a local file to S3 and return the download URL.

    Args:
        filepath: Path relative to the working directory (e.g. 'artifacts/report.pdf').

    Returns:
        The download URL, or an error message.
    """
    logger.info(f"###### upload_file_to_s3: {filepath} ######")
    try:
        import boto3
        from urllib import parse as url_parse

        s3_bucket = config.get("s3_bucket")
        if not s3_bucket:
            return "S3 bucket is not configured."

        full_path = os.path.join(WORKING_DIR, filepath)
        if not os.path.exists(full_path):
            return f"File not found: {filepath}"

        content_type = utils.get_contents_type(filepath)
        s3 = boto3.client("s3", region_name=config.get("region", "us-west-2"))

        with open(full_path, "rb") as f:
            s3.put_object(Bucket=s3_bucket, Key=filepath, Body=f.read(), ContentType=content_type)

        if sharing_url:
            url = f"{sharing_url}/{url_parse.quote(filepath)}"
            return f"Upload complete: {url}"
        return f"Upload complete: {s3_uri_to_console_url(f"s3://{s3_bucket}/{filepath}", config.get("region", "us-west-2"))}"

    except Exception as e:
        return f"Upload failed: {str(e)}"

@function_tool
def memory_search(query: str, max_results: int = 5, min_score: float = 0.0) -> str:
    """Search across memory files (MEMORY.md and memory/*.md) for relevant information.

    Performs keyword-based search over all memory files and returns matching snippets
    ranked by relevance score.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default: 5).
        min_score: Minimum relevance score threshold 0.0-1.0 (default: 0.0).

    Returns:
        JSON array of matching snippets with text, path, from (line), lines, and score.
    """
    import re as _re
    logger.info(f"###### memory_search: {query} ######")

    memory_root = Path(WORKING_DIR)
    memory_dir = memory_root / "memory"

    target_files = []
    memory_md = memory_root / "MEMORY.md"
    if memory_md.exists():
        target_files.append(memory_md)
    if memory_dir.exists():
        target_files.extend(sorted(memory_dir.glob("*.md"), reverse=True))

    if not target_files:
        return json.dumps([], ensure_ascii=False)

    query_lower = query.lower()
    query_tokens = [t for t in _re.split(r'\s+', query_lower) if len(t) >= 2]

    results = []
    for fpath in target_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.split("\n")
        content_lower = content.lower()

        if not any(tok in content_lower for tok in query_tokens):
            continue

        window_size = 5
        for i in range(0, len(lines), window_size):
            chunk_lines = lines[i:i + window_size]
            chunk_text = "\n".join(chunk_lines)
            chunk_lower = chunk_text.lower()

            matched_tokens = sum(1 for tok in query_tokens if tok in chunk_lower)
            if matched_tokens == 0:
                continue

            score = matched_tokens / len(query_tokens) if query_tokens else 0.0

            if score >= min_score:
                rel_path = str(fpath.relative_to(memory_root))
                results.append({
                    "text": chunk_text.strip(),
                    "path": rel_path,
                    "from": i + 1,
                    "lines": len(chunk_lines),
                    "score": round(score, 3),
                })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:max_results]

    return json.dumps(results, indent=2, ensure_ascii=False)


@function_tool
def memory_get(path: str, from_line: int = 0, lines: int = 0) -> str:
    """Read a specific memory file (MEMORY.md or memory/*.md).

    Use after memory_search to get full context, or when you know the exact file path.

    Args:
        path: Workspace-relative path (e.g. "MEMORY.md", "memory/2026-03-02.md").
        from_line: Starting line number, 1-indexed (0 = read from beginning).
        lines: Number of lines to read (0 = read entire file).

    Returns:
        JSON with 'text' (file content) and 'path'. Returns empty text if file doesn't exist.
    """
    logger.info(f"###### memory_get: {path} ######")

    full_path = Path(WORKING_DIR) / path

    if not full_path.exists():
        return json.dumps({"text": "", "path": path}, ensure_ascii=False)

    try:
        content = full_path.read_text(encoding="utf-8")

        if from_line > 0 or lines > 0:
            all_lines = content.split("\n")
            start = max(0, from_line - 1)
            if lines > 0:
                end = start + lines
                content = "\n".join(all_lines[start:end])
            else:
                content = "\n".join(all_lines[start:])

        return json.dumps({"text": content, "path": path}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"text": f"Error reading file: {e}", "path": path}, ensure_ascii=False)


@function_tool
def get_skill_instructions(plugin_name: str, skill_name: str) -> str:
    """Load the full instructions for a specific skill by name.

    Use this when you need detailed instructions for a task that matches
    one of the available skills listed in the system prompt.

    Args:
        plugin_name: The plugin name (e.g. 'base', 'frontend-design').
        skill_name: The name of the skill to load (e.g. 'pdf').

    Returns:
        The full skill instructions, or an error message if not found.
    """
    logger.info(f"###### get_skill_instructions: {skill_name} (plugin={plugin_name}) ######")
    skill_mgr = skill.skill_managers.get(plugin_name)
    if skill_mgr is None:
        if plugin_name == "base":
            skills_dir = skill.SKILLS_DIR
        else:
            skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")
        skill_mgr = skill.SkillManager(skills_dir)
        skill.skill_managers[plugin_name] = skill_mgr

    instructions = skill_mgr.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    base_mgr = skill.skill_managers.get("base")
    if base_mgr is None:
        base_mgr = skill.SkillManager(skill.SKILLS_DIR)
        skill.skill_managers["base"] = base_mgr
    instructions = base_mgr.get_skill_instructions(skill_name)
    if instructions:
        return instructions

    available = ", ".join(skill_mgr.registry.keys())
    return f"Skill '{skill_name}' not found. Available skills: {available}"

def _ensure_cli_scripts_on_path() -> None:
    """Prepend pip user script dir so CLIs (e.g. browser-use) resolve in subprocess."""
    import site
    import sysconfig

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
    path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    for d in reversed(extra):
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


@function_tool
def bash(command: str) -> str:
    """Execute a bash command and return the result"""
    logger.info(f"###### bash: {command} ######")
    _ensure_cli_scripts_on_path()
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=WORKING_DIR, timeout=300,
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

def get_builtin_tools() -> list:
    """Return the list of built-in tools for the skill-aware agent."""
    return [execute_code, bash, upload_file_to_s3]

#########################################################
# OpenAI Agents SDK
#########################################################

_bedrock_configured = False
_last_bedrock_region: str | None = None
_last_model_id: str | None = None

agent = None
selected_agent_tools: list[str] = []
selected_mcp_servers: list[str] = []
selected_skill_list: list[str] = []
_conversations_sessions: dict[str, ConversationsSession] = {}


def get_conversations_session(user_key: str | None = None) -> ConversationsSession:
    """Return a ConversationsSession for the current chat user."""
    key = user_key or chat.user_id or "default"
    if key not in _conversations_sessions:
        _conversations_sessions[key] = ConversationsSession(
            db_path=CONVERSATIONS_DB,
            conversations_dir=CONVERSATIONS_DIR,
            user_key=key,
            session_settings=SessionSettings(limit=CONVERSATION_HISTORY_LIMIT),
        )
        logger.info(f"ConversationsSession ready for user_key={key}")
    return _conversations_sessions[key]


async def clear_conversations_session(user_key: str | None = None) -> None:
    """Clear conversation history for the given user."""
    key = user_key or chat.user_id or "default"
    session = _conversations_sessions.pop(key, None) or get_conversations_session(key)
    await session.clear_session()
    _conversations_sessions.pop(key, None)
    logger.info(f"ConversationsSession cleared for user_key={key}")


def _get_agent_model_id() -> str:
    return chat.get_responses_model_id()


def _get_agent_region() -> str:
    return chat.get_responses_region()


def _configure_bedrock_client() -> AsyncBedrockOpenAI:
    global _bedrock_configured, _last_bedrock_region, _last_model_id
    region = _get_agent_region()
    model = _get_agent_model_id()
    client = AsyncBedrockOpenAI(
        aws_region=region,
        bedrock_token_provider=lambda r=region: provide_token(region=r),
    )
    set_default_openai_client(client)
    _bedrock_configured = True
    _last_bedrock_region = region
    _last_model_id = model
    logger.info(f"Bedrock OpenAI client configured (model={model}, region={region})")
    return client


def _build_mcp_servers(server_names: list[str]) -> list[Any]:
    servers: list[Any] = []
    for tool_name in server_names:
        config = mcp_config.load_config(tool_name)
        if not config or "mcpServers" not in config:
            logger.warning(f"No MCP config for: {tool_name}")
            continue

        server_key = next(iter(config["mcpServers"]))
        server_config = config["mcpServers"][server_key]

        if server_config.get("type") == "streamable_http":
            url = server_config["url"]
            headers = server_config.get("headers", {})
            logger.info(f"MCP streamable_http: {tool_name} -> {url}")
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
            logger.warning(f"Skipping MCP {tool_name}: executable not found at {cmd_path}")
            continue

        logger.info(f"MCP stdio: {tool_name} command={command} args={args}")
        servers.append(
            MCPServerStdio(
                params={"command": command, "args": args, "env": env},
                name=tool_name,
                cache_tools_list=True,
            )
        )

    return servers


@function_tool
def builtin_current_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@function_tool
def builtin_file_read(path: str, from_line: int = 0, lines: int = 0) -> str:
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
def builtin_file_write(path: str, content: str) -> str:
    """Write text to a file under the application working directory."""
    full_path = Path(WORKING_DIR) / path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return json.dumps({"path": path, "status": "ok"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "path": path}, ensure_ascii=False)


OPTIONAL_TOOL_MAP = {
    "current_time": builtin_current_time,
    "file_read": builtin_file_read,
    "file_write": builtin_file_write,
}


def _resolve_optional_tools(tool_names: list[str]) -> list[Any]:
    tools = []
    for name in tool_names:
        fn = OPTIONAL_TOOL_MAP.get(name)
        if fn:
            tools.append(fn)
        else:
            logger.warning(f"Unknown optional tool: {name}")
    return tools


def _build_instructions(skill_list: list[str]) -> str:
    if chat.skill_mode == "Enable":
        skill_info = skill.get_skill_info(skill_list)
        logger.info(f"skill_info: {skill_info}")
        return skill.build_skill_prompt(skill_info)
    return BASE_SYSTEM_PROMPT


def create_agent(
    optional_tool_names: list[str],
    mcp_server_names: list[str],
    skill_list: list[str],
):
    """Create an OpenAI Agents SDK Agent backed by Bedrock OpenAI."""
    _configure_bedrock_client()
    builtin = get_builtin_tools()
    if chat.skill_mode == "Enable":
        builtin = builtin + [get_skill_instructions]

    tools = list(builtin) + _resolve_optional_tools(optional_tool_names)
    mcp_servers = _build_mcp_servers(mcp_server_names)
    model = _get_agent_model_id()
    logger.info(f"Creating Agent model={model} tools={len(tools)} mcp={len(mcp_servers)}")

    return Agent(
        name="서연",
        instructions=_build_instructions(skill_list),
        model=model,
        tools=tools,
        mcp_servers=mcp_servers,
    )


def _text_from_run_item(event: RunItemStreamEvent) -> Optional[str]:
    item = event.item
    if hasattr(item, "raw_item") and item.raw_item is not None:
        raw = item.raw_item
        if hasattr(raw, "content") and raw.content:
            for block in raw.content:
                if hasattr(block, "text") and block.text:
                    return block.text
    return None


async def _consume_agent_stream(
    agent: Agent,
    query: str,
    notification_queue: Any,
    session: ConversationsSession,
) -> tuple[str, list[str], list[dict], str]:
    queue = notification_queue
    image_urls: list[str] = []
    references: list[dict] = []
    current = ""
    final_result = ""

    history = await session.get_items()
    logger.info(
        f"ConversationsSession user={session._user_key} "
        f"conversation_id={session.session_id} history_items={len(history)}"
    )

    result = Runner.run_streamed(agent, query, max_turns=20, session=session)

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
                if raw is not None:
                    tool_name = getattr(raw, "name", "") or ""
                    tool_use_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or tool_name
                    args = getattr(raw, "arguments", "")
                    if queue is not None:
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
                    _info_content, urls, refs = chat.get_tool_info(tool_name, str(output))
                    references.extend(refs or [])
                    image_urls.extend(urls or [])

            elif event.name == "message_output_created":
                text = _text_from_run_item(event)
                if text:
                    final_result = text
    except ModelBehaviorError as exc:
        logger.error(f"Agent model error: {exc}")
        if queue is not None:
            queue.notify(f"Model error: {exc}")
        if not final_result and not current:
            final_result = (
                "Bedrock API 처리 중 서버 오류가 발생했습니다. "
                "대화가 길거나 도구 결과가 많으면 **대화 초기화** 후 다시 시도해 주세요."
            )
    except Exception as exc:
        logger.error(f"Agent stream error: {exc}")
        if queue is not None:
            queue.notify(f"Agent error: {exc}")
        if not final_result and not current:
            raise

    if not final_result and current:
        final_result = current

    return final_result, image_urls, references, current


async def _stream_agent_to_queue(
    agent: Agent,
    query: str,
    notification_queue: Any,
    session: ConversationsSession,
) -> tuple[str, list[str]]:
    queue = notification_queue
    if queue is not None:
        queue.reset()

    mcp_servers = list(agent.mcp_servers)

    async def _run_connected() -> tuple[str, list[str], list[dict], str]:
        return await _consume_agent_stream(agent, query, notification_queue, session)

    if mcp_servers:
        async with MCPServerManager(
            mcp_servers,
            drop_failed_servers=True,
            strict=False,
        ) as manager:
            agent.mcp_servers = manager.active_servers
            if manager.failed_servers:
                for server, err in manager.errors.items():
                    logger.warning(
                        f"MCP server '{getattr(server, 'name', server)}' failed to connect: {err}"
                    )
            if mcp_servers and not manager.active_servers:
                raise RuntimeError(
                    "No MCP servers connected. Check MCP configuration and dependencies."
                )
            final_result, image_urls, references, _current = await _run_connected()
    else:
        final_result, image_urls, references, _current = await _run_connected()

    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            content = reference["content"][:100].replace("\n", "")
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {content}...\n"
        final_result += ref

    if queue is not None:
        queue.result(final_result)

    return final_result, image_urls


async def run_agent(
    query: str,
    optional_tool_names: list[str],
    mcp_server_names: list[str],
    skill_list: list[str],
    notification_queue,
):
    """Run the OpenAI Agents SDK agent with streaming and tool notifications."""
    global agent, selected_agent_tools, selected_mcp_servers, selected_skill_list

    current_region = _get_agent_region()
    current_model = _get_agent_model_id()
    if (
        selected_agent_tools != optional_tool_names
        or selected_mcp_servers != mcp_server_names
        or selected_skill_list != skill_list
        or agent is None
        or _last_bedrock_region != current_region
        or _last_model_id != current_model
    ):
        selected_agent_tools = list(optional_tool_names)
        selected_mcp_servers = list(mcp_server_names)
        selected_skill_list = list(skill_list)
        agent = create_agent(optional_tool_names, mcp_server_names, skill_list)
        logger.info("Agent recreated (config changed)")

    session = get_conversations_session()
    return await _stream_agent_to_queue(agent, query, notification_queue, session)


