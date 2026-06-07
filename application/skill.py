"""OpenAI Agents SDK Skills pattern for local Agent runs.

Follows the Skills capability model:
https://openai.github.io/openai-agents-python/ref/sandbox/capabilities/skills/

Skills are indexed by metadata only in the system prompt. The agent calls
``load_skill`` to materialize a skill under ``.agents/<name>/``, then reads
``SKILL.md`` and related files with ``file_read`` / ``bash``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("skill")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(WORKING_DIR, "skills")
ARTIFACTS_DIR = os.path.join(WORKING_DIR, "artifacts")
SKILLS_WORKSPACE = ".agents"

config = None


def _load_config():
    global config
    if config is None:
        import utils

        config = utils.load_config()
    return config


_SKILLS_SECTION_INTRO = (
    "A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. "
    "Below is the list of skills that can be used. Each entry includes a name, description, "
    "and file path so you can open the source for full instructions when using a specific skill."
)

_HOW_TO_USE_LAZY_SKILLS_SECTION = "\n".join(
    [
        "### How to use skills",
        "- Discovery: The list above is the skill index available in this session "
        "(name + description + workspace path). In lazy mode, those paths are loaded "
        "on demand instead of being present up front.",
        "- Trigger rules: If the user names a skill (with `$SkillName` or plain text) "
        "OR the task clearly matches a skill's description shown above, you must use that "
        "skill for that turn. Multiple mentions mean use them all. Do not carry skills "
        "across turns unless re-mentioned.",
        "- Missing/blocked: If a named skill isn't in the list or the path can't be read, "
        "say so briefly and continue with the best fallback.",
        "- How to use a skill (progressive disclosure):",
        "  1) After deciding to use a lazy skill, call `load_skill` for that skill first, "
        "then open its `SKILL.md` with `file_read`.",
        "  2) If `SKILL.md` points to extra folders such as `references/`, load only the "
        "specific files needed for the request; don't bulk-load everything.",
        "  3) If `scripts/` exist, prefer running or patching them instead of retyping "
        "large code blocks.",
        "  4) If `assets/` or templates exist, reuse them instead of recreating from scratch.",
        "- Coordination and sequencing:",
        "  - If multiple skills apply, choose the minimal set that covers the request "
        "and state the order you'll use them.",
        "  - Announce which skill(s) you're using and why (one short line). "
        "If you skip an obvious skill, say why.",
        "- Context hygiene:",
        "  - Keep context small: summarize long sections instead of pasting them; "
        "only load extra files when needed.",
        "  - Avoid deep reference-chasing: prefer opening only files directly linked "
        "from `SKILL.md` unless you're blocked.",
        "  - When variants exist (frameworks, providers, domains), pick only the relevant "
        "reference file(s) and note that choice.",
        "- Safety and fallback: If a skill can't be applied cleanly (missing files, "
        "unclear instructions), state the issue, pick the next-best approach, and continue.",
    ]
)


@dataclass(frozen=True)
class SkillMetadata:
    """Indexed metadata for a skill (OpenAI Agents SDK SkillMetadata)."""

    name: str
    description: str
    path: Path
    source_dir: Path


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: str


class SkillManager:
    """Discover skills from local directories (SKILL.md frontmatter)."""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.registry: dict[str, Skill] = {}
        self._discover(skills_dir)

    def _discover(self, skills_dir: str) -> None:
        if not os.path.isdir(skills_dir):
            logger.info(f"skills directory is not found: {skills_dir}")
            return

        for entry in os.listdir(skills_dir):
            skill_md = os.path.join(skills_dir, entry, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            try:
                meta, instructions = self._parse_skill_md(skill_md)
                skill_obj = Skill(
                    name=meta.get("name", entry),
                    description=meta.get("description", ""),
                    instructions=instructions,
                    path=os.path.join(skills_dir, entry),
                )
                self.registry[skill_obj.name] = skill_obj
                logger.info(f"Skill discovered: {skill_obj.name}")
            except Exception as exc:
                logger.warning(f"Failed to load skill '{entry}': {exc}")

    def discover_plugin_skills(self, skills_dir: str) -> None:
        if not os.path.isdir(skills_dir):
            return
        self._discover(skills_dir)

    @staticmethod
    def _parse_skill_md(filepath: str) -> tuple[dict, str]:
        with open(filepath, "r", encoding="utf-8") as handle:
            raw = handle.read()

        if not raw.startswith("---"):
            return {}, raw

        parts = raw.split("---", 2)
        if len(parts) < 3:
            return {}, raw

        frontmatter = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return frontmatter, body

    def list_skill_metadata(self, skills_path: str = SKILLS_WORKSPACE) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for skill_obj in sorted(self.registry.values(), key=lambda item: item.name):
            dir_name = os.path.basename(skill_obj.path.rstrip(os.sep))
            metadata.append(
                SkillMetadata(
                    name=skill_obj.name,
                    description=skill_obj.description or "No description provided.",
                    path=Path(skills_path) / dir_name,
                    source_dir=Path(skill_obj.path),
                )
            )
        return metadata


skill_managers: dict[str, SkillManager] = {}


def _get_skill_manager(plugin_name: str = "base") -> SkillManager:
    manager = skill_managers.get(plugin_name)
    if manager is not None:
        return manager

    if plugin_name == "base":
        skills_dir = SKILLS_DIR
    else:
        skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")

    manager = SkillManager(skills_dir)
    skill_managers[plugin_name] = manager
    return manager


def register_plugin_skills(plugin_name: str) -> None:
    if plugin_name == "base":
        skills_dir = SKILLS_DIR
    else:
        skills_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "skills")

    manager = skill_managers.get(plugin_name)
    if manager is None:
        manager = SkillManager(skills_dir)
        skill_managers[plugin_name] = manager
    else:
        manager.discover_plugin_skills(skills_dir)


def available_skill_info(plugin_name: str = "base") -> list[dict[str, str]]:
    manager = _get_skill_manager(plugin_name)
    return [{"name": item.name, "description": item.description} for item in manager.registry.values()]


def get_skill_info(skill_list: list[str]) -> list[dict[str, str]]:
    manager = _get_skill_manager("base")
    selected = []
    for item in manager.registry.values():
        if item.name in skill_list:
            selected.append({"name": item.name, "description": item.description})
    return selected


def get_plugin_skill_info(plugin_name: str, plugin_skill_list: list[str]) -> list[dict[str, str]]:
    manager = _get_skill_manager(plugin_name)
    selected = []
    for item in manager.registry.values():
        if item.name in plugin_skill_list:
            selected.append({"name": item.name, "description": item.description})
    logger.info(f"plugin_skill_info: {selected}")
    return selected


def selected_skill_info(plugin_name: str) -> list[dict[str, str]]:
    cfg = _load_config()
    if plugin_name == "base":
        skill_list = cfg.get("default_skills") or []
    else:
        skill_list = cfg.get("plugin_skills", {}).get(plugin_name) or []
    logger.info(f"plugin_name: {plugin_name}, skill_list: {skill_list}")

    return [item for item in available_skill_info(plugin_name) if item["name"] in skill_list]


def get_selected_skill_metadata(skill_list: list[str]) -> list[SkillMetadata]:
    """Return OpenAI-style metadata for UI-selected skills."""
    all_metadata = _get_skill_manager("base").list_skill_metadata()
    if not skill_list:
        return all_metadata
    allowed = set(skill_list)
    return [item for item in all_metadata if item.name in allowed]


def _find_skill_metadata(skill_name: str) -> SkillMetadata | None:
    normalized = skill_name.strip()
    for item in _get_skill_manager("base").list_skill_metadata():
        if item.name == normalized or item.path.name == normalized:
            return item
    return None


def load_skill(skill_name: str) -> dict[str, str]:
    """Materialize one skill under ``.agents/<name>/`` (lazy loading)."""
    logger.info(f"load_skill: {skill_name}")
    metadata = _find_skill_metadata(skill_name)
    if metadata is None:
        available = ", ".join(item.name for item in _get_skill_manager("base").list_skill_metadata())
        return {
            "status": "error",
            "skill_name": skill_name,
            "message": f"Skill '{skill_name}' not found. Available skills: {available}",
        }

    workspace_root = Path(WORKING_DIR)
    skill_dest = workspace_root / metadata.path
    skill_md = skill_dest / "SKILL.md"

    if skill_md.is_file():
        return {
            "status": "already_loaded",
            "skill_name": metadata.name,
            "path": metadata.path.as_posix(),
        }

    if not metadata.source_dir.is_dir():
        return {
            "status": "error",
            "skill_name": metadata.name,
            "message": f"Skill source directory not found: {metadata.source_dir}",
        }

    skill_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(metadata.source_dir, skill_dest, dirs_exist_ok=True)
    logger.info(f"Skill loaded: {metadata.name} -> {skill_dest}")

    return {
        "status": "loaded",
        "skill_name": metadata.name,
        "path": metadata.path.as_posix(),
    }


def build_path_info() -> str:
    return (
        f"## Paths\n"
        f"- WORKING_DIR: {WORKING_DIR}\n"
        f"- ARTIFACTS_DIR: {ARTIFACTS_DIR}\n"
        f"- SKILLS_WORKSPACE: {os.path.join(WORKING_DIR, SKILLS_WORKSPACE)}\n"
        f"Example: file_read(path='{SKILLS_WORKSPACE}/docx/SKILL.md')\n"
    )


def build_skills_instructions(
    metadata: list[SkillMetadata],
    *,
    lazy: bool = True,
) -> str | None:
    """Build the OpenAI Skills instructions block for the agent system prompt."""
    if not metadata:
        return None

    available_lines = [
        f"- {item.name}: {item.description} (file: {item.path.as_posix()})"
        for item in metadata
    ]

    sections = [
        "## Skills",
        _SKILLS_SECTION_INTRO,
        "### Available skills",
        *available_lines,
    ]

    if lazy:
        sections.extend(
            [
                "### Lazy loading",
                "- These skills are indexed for planning, but they are not materialized "
                "in the workspace yet.",
                "- Call `load_skill` with a single skill name from the list before "
                "reading its `SKILL.md` or other files from the workspace.",
                "- `load_skill` stages exactly one skill under the listed path. "
                "If you need more than one skill, call it multiple times.",
            ]
        )

    sections.append(_HOW_TO_USE_LAZY_SKILLS_SECTION if lazy else _HOW_TO_USE_LAZY_SKILLS_SECTION)
    return "\n".join(sections)


AGENT_BASE_PROMPT = (
    "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다.\n"
    "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다.\n"
    "모르는 질문을 받으면 솔직히 모른다고 말합니다.\n"
    "한국어로 답변하세요.\n\n"
    "## Agent Workflow\n"
    "1. 사용자 입력을 받는다\n"
    "2. 요청에 맞는 skill이 있으면 `load_skill`로 워크스페이스에 로드한 뒤 `file_read`로 SKILL.md를 연다\n"
    "3. skill 지침에 따라 execute_code, file_write, bash 등의 도구로 작업을 수행한다\n"
    "   (생성 파일·스크립트는 `artifacts/` 아래에 저장한다)\n"
    "4. 결과 파일이 있으면 upload_file_to_s3로 업로드하여 URL을 제공한다\n"
    "5. 최종 결과를 사용자에게 전달한다\n"
)


def build_agent_instructions(selected_skills: list[str]) -> str:
    """Assemble the full agent system prompt with OpenAI Skills metadata."""
    metadata = get_selected_skill_metadata(selected_skills)
    skills_block = build_skills_instructions(metadata, lazy=True)
    parts = [AGENT_BASE_PROMPT, build_path_info()]
    if skills_block:
        parts.append(skills_block)
    return "\n\n".join(parts)


def get_command_instructions(plugin_name: str, command_name: str) -> str:
    logger.info(f"get_command_instructions: {command_name}")

    commands_dir = os.path.join(WORKING_DIR, "plugins", plugin_name, "commands")
    if not os.path.isdir(commands_dir):
        return f"Plugin '{plugin_name}' has no commands directory."

    command_name_normalized = command_name.lower().strip()
    filepath = os.path.join(commands_dir, f"{command_name_normalized}.md")

    if not os.path.isfile(filepath):
        available = [name[:-3] for name in os.listdir(commands_dir) if name.endswith(".md")]
        return f"Command '{command_name}' not found. Available commands: {', '.join(available)}"

    frontmatter, body = SkillManager._parse_skill_md(filepath)
    if frontmatter:
        desc = frontmatter.get("description", "")
        hint = frontmatter.get("argument-hint", "")
        header = f"**{desc}**\n"
        if hint:
            header += f"Argument hint: {hint}\n\n"
        return header + body
    return body


def build_command_prompt(plugin_name: str, skill_list: list[str], command: str) -> str:
    command_instructions = get_command_instructions(plugin_name, command)
    command_section = (
        f"## Command Instructions\n<command_instructions>\n{command_instructions}\n</command_instructions>\n"
    )
    skills_prompt = build_agent_instructions(skill_list)
    return f"{skills_prompt}\n\n{command_section}"
