"""대화 세션 저장소 (서버 메모리).

서버를 재시작하면 대화가 사라진다. 클라이언트(브라우저)마다 대화를 분리해서
다른 사람의 대화는 조회할 수 없다. 세션 수와 세션당 메시지 수에 상한이 있고,
한 대화에서는 한 번에 하나의 답변만 생성한다.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from chatbot.model import Message

TITLE_CHARS = 30


@dataclass
class Session:
    id: str
    client_id: str
    model: str
    title: str = "새 대화"
    messages: list[Message] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    busy_since: float | None = None  # 답변 생성 중이면 시작 시각


class SessionNotFound(KeyError):
    pass


class SessionBusy(Exception):
    """같은 대화에서 이미 답변을 생성 중일 때."""


class ConversationStore:
    def __init__(
        self,
        *,
        max_per_client: int = 30,
        max_total: int = 500,
        ttl_seconds: float = 24 * 3600,
        max_messages_per_session: int = 200,
        busy_timeout_seconds: float = 300,
        clock: Callable[[], float] = time.time,
    ):
        self.max_per_client = max_per_client
        self.max_total = max_total
        self.ttl_seconds = ttl_seconds
        # user+assistant 쌍 단위로 자르므로 짝수로 맞춘다
        self.max_messages_per_session = max(2, max_messages_per_session - max_messages_per_session % 2)
        self.busy_timeout_seconds = busy_timeout_seconds
        self.clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, client_id: str, model: str) -> Session:
        now = self.clock()
        session = Session(id=uuid.uuid4().hex, client_id=client_id, model=model, created_at=now, updated_at=now)
        with self._lock:
            self._cleanup_locked()
            self._sessions[session.id] = session
            self._enforce_limits_locked(client_id)
        return session

    def get(self, client_id: str, session_id: str) -> Session:
        """자기 대화만 조회 가능. 남의 대화거나 없으면 SessionNotFound."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.client_id != client_id or self._expired(session):
                raise SessionNotFound(session_id)
            return session

    def list(self, client_id: str) -> list[Session]:
        with self._lock:
            self._cleanup_locked()
            mine = [s for s in self._sessions.values() if s.client_id == client_id]
        return sorted(mine, key=lambda s: s.updated_at, reverse=True)

    def add_exchange(self, session: Session, user_message: str, reply: str) -> None:
        """질문과 답변을 한 쌍으로 저장한다 (답변 실패 시엔 호출하지 않음)."""
        with self._lock:
            if not session.messages:
                session.title = user_message[:TITLE_CHARS] + ("…" if len(user_message) > TITLE_CHARS else "")
            session.messages.append({"role": "user", "content": user_message})
            session.messages.append({"role": "assistant", "content": reply})
            overflow = len(session.messages) - self.max_messages_per_session
            if overflow > 0:
                del session.messages[:overflow]  # 오래된 턴부터 (항상 짝수라 user 로 시작)
            session.updated_at = self.clock()

    def acquire(self, session: Session) -> None:
        """답변 생성 시작. 이미 생성 중이면 SessionBusy.

        생성 시작부터 저장까지를 한 단위로 묶어서, 두 탭에서 동시에 보내도
        서로의 질문을 못 본 채 답하는 일이 없게 한다. 오래된 잠금(연결이 끊겨
        풀리지 않은 경우)은 busy_timeout_seconds 뒤에 무시한다.
        """
        with self._lock:
            now = self.clock()
            if session.busy_since is not None and now - session.busy_since < self.busy_timeout_seconds:
                raise SessionBusy(session.id)
            session.busy_since = now

    def release(self, session: Session) -> None:
        with self._lock:
            session.busy_since = None

    def delete(self, client_id: str, session_id: str) -> None:
        self.get(client_id, session_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    def _expired(self, session: Session) -> bool:
        return self.clock() - session.updated_at > self.ttl_seconds

    def _cleanup_locked(self) -> None:
        for sid in [sid for sid, s in self._sessions.items() if self._expired(s)]:
            del self._sessions[sid]

    def _enforce_limits_locked(self, client_id: str) -> None:
        # 클라이언트별, 전체 개수를 넘으면 가장 오래 안 쓴 대화부터 삭제
        mine = sorted((s for s in self._sessions.values() if s.client_id == client_id), key=lambda s: s.updated_at)
        for s in mine[: max(0, len(mine) - self.max_per_client)]:
            del self._sessions[s.id]
        overflow = len(self._sessions) - self.max_total
        if overflow > 0:
            for s in sorted(self._sessions.values(), key=lambda s: s.updated_at)[:overflow]:
                del self._sessions[s.id]
