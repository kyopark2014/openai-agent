"""Local ConversationsSession compatible with OpenAI Agents SDK Session protocol.

Bedrock OpenAI does not expose the OpenAI Conversations API, so this module
implements the same interface as ``OpenAIConversationsSession`` using SQLite
storage keyed by ``conversation_id``.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from agents.items import TResponseInputItem
from agents.memory.session import SessionABC
from agents.memory.session_settings import SessionSettings, resolve_session_limit
from agents.memory.sqlite_session import SQLiteSession

logger = logging.getLogger("conversations-session")


def _default_db_path() -> Path:
    return Path(__file__).resolve().parent / ".conversations.db"


def _conversation_meta_path(conversations_dir: Path, user_key: str) -> Path:
    safe_key = user_key.replace("/", "_")
    return conversations_dir / f"{safe_key}.json"


class ConversationsSession(SessionABC):
    """Session storage modeled after OpenAIConversationsSession.

    Args:
        conversation_id: Existing conversation id. When omitted, a new id is
            created on first use.
        db_path: SQLite database path for message storage.
        conversations_dir: Directory for per-user conversation metadata.
        user_key: Logical owner key (e.g. chat user id) used to persist the
            conversation id across process restarts.
        session_settings: Optional retrieval limits for history.
    """

    session_settings: SessionSettings | None = None

    def __init__(
        self,
        *,
        conversation_id: str | None = None,
        db_path: str | Path | None = None,
        conversations_dir: str | Path | None = None,
        user_key: str | None = None,
        session_settings: SessionSettings | None = None,
    ):
        self._conversation_id: str | None = conversation_id
        self._db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._conversations_dir = (
            Path(conversations_dir)
            if conversations_dir is not None
            else self._db_path.parent / ".conversations"
        )
        self._user_key = user_key
        self.session_settings = session_settings or SessionSettings()
        self._store: SQLiteSession | None = None

    @property
    def session_id(self) -> str:
        """Conversation id for this session."""
        if self._conversation_id is None:
            raise ValueError(
                "Session ID not yet available. The session is lazily initialized "
                "on first API call. Call get_items(), add_items(), or similar first."
            )
        return self._conversation_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._conversation_id = value
        self._store = None
        if self._user_key:
            self._save_conversation_id(value)

    async def _get_conversation_id(self) -> str:
        if self._conversation_id is None:
            if self._user_key:
                stored = self._load_conversation_id(self._user_key)
                if stored:
                    self._conversation_id = stored
            if self._conversation_id is None:
                self._conversation_id = f"conv_{uuid.uuid4().hex}"
                if self._user_key:
                    self._save_conversation_id(self._conversation_id)
                logger.info(
                    "ConversationsSession created conversation_id=%s user_key=%s",
                    self._conversation_id,
                    self._user_key,
                )
        return self._conversation_id

    async def _clear_conversation_id(self) -> None:
        if self._user_key:
            meta_path = _conversation_meta_path(self._conversations_dir, self._user_key)
            if meta_path.exists():
                meta_path.unlink()
        self._conversation_id = None
        self._store = None

    def _load_conversation_id(self, user_key: str) -> str | None:
        meta_path = _conversation_meta_path(self._conversations_dir, user_key)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            conversation_id = data.get("conversation_id")
            return str(conversation_id) if conversation_id else None
        except Exception as exc:
            logger.warning("Failed to load conversation metadata %s: %s", meta_path, exc)
            return None

    def _save_conversation_id(self, conversation_id: str) -> None:
        if not self._user_key:
            return
        self._conversations_dir.mkdir(parents=True, exist_ok=True)
        meta_path = _conversation_meta_path(self._conversations_dir, self._user_key)
        meta_path.write_text(
            json.dumps({"conversation_id": conversation_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _get_store(self) -> SQLiteSession:
        conversation_id = await self._get_conversation_id()
        if self._store is None or self._store.session_id != conversation_id:
            self._store = SQLiteSession(
                session_id=conversation_id,
                db_path=self._db_path,
                session_settings=self.session_settings,
            )
        return self._store

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        store = await self._get_store()
        session_limit = resolve_session_limit(limit, self.session_settings)
        return await store.get_items(limit=session_limit)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if not items:
            return
        store = await self._get_store()
        await store.add_items(items)

    async def pop_item(self) -> TResponseInputItem | None:
        store = await self._get_store()
        return await store.pop_item()

    async def clear_session(self) -> None:
        if self._conversation_id is not None:
            store = await self._get_store()
            await store.clear_session()
        await self._clear_conversation_id()
