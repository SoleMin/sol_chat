"""답변 생성 흐름: 전처리 -> 모델 백엔드 -> 후처리.

CLI 와 웹 서버가 모두 이 클래스를 통해 답을 만든다.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

from chatbot.config import ChatConfig, ConfigError
from chatbot.model import BackendError, ChatBackend, Message, create_backend
from chatbot.postprocessing import clean_chunk, clean_response
from chatbot.preprocessing import build_messages


class ChatGenerator:
    def __init__(
        self,
        config: ChatConfig,
        backend_factory: Callable[..., ChatBackend] = create_backend,
    ):
        self.config = config
        self.backend_factory = backend_factory
        self._backends: dict[str, ChatBackend] = {}
        self._locks: dict[str, threading.Lock] = {}  # 모델별 초기화 잠금
        self._locks_guard = threading.Lock()

    def backend(self, model_name: str | None) -> ChatBackend:
        """모델별 백엔드를 처음 쓸 때 만들고 재사용한다 (키 없는 모델이 있어도 서버는 뜸).

        첫 요청이 동시에 와도 같은 모델은 한 번만 만든다 (로컬 모델 중복 로드 방지).
        다른 모델끼리는 서로 기다리지 않고, 만들다 실패하면 다음 요청에서 다시 시도한다.
        """
        try:
            entry = self.config.get_model(model_name)
        except ConfigError as e:
            raise BackendError(str(e), status_code=400) from e
        backend = self._backends.get(entry.name)
        if backend is not None:
            return backend
        with self._locks_guard:
            lock = self._locks.setdefault(entry.name, threading.Lock())
        with lock:
            if entry.name not in self._backends:
                self._backends[entry.name] = self.backend_factory(entry)
            return self._backends[entry.name]

    def _messages(self, history: list[Message], user_message: str) -> list[Message]:
        return build_messages(
            self.config.system_prompt,
            history,
            user_message,
            max_turns=self.config.history.max_turns,
            max_chars=self.config.history.max_chars,
        )

    def reply(self, model_name: str | None, history: list[Message], user_message: str) -> str:
        backend = self.backend(model_name)
        text = backend.generate(self._messages(history, user_message), self.config.generation)
        return clean_response(text)

    def stream_reply(self, model_name: str | None, history: list[Message], user_message: str) -> Iterator[str]:
        backend = self.backend(model_name)
        for chunk in backend.stream(self._messages(history, user_message), self.config.generation):
            chunk = clean_chunk(chunk)
            if chunk:
                yield chunk
