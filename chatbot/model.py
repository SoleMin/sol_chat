"""모델 백엔드 인터페이스와 백엔드 선택(팩토리).

챗봇의 나머지 코드는 "messages 를 주면 답을 주는 것"만 알고,
그게 OpenRouter API 인지, 내가 학습한 LoRA 모델인지는 모른다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from chatbot.config import ConfigError, GenerationParams, ModelEntry

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class BackendError(Exception):
    """모델 호출 실패. user_message 는 화면에 그대로 보여줘도 되는 설명."""

    def __init__(self, user_message: str, *, status_code: int = 502):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class ChatBackend(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], params: GenerationParams) -> str:
        """답변 전체를 한 번에 돌려준다."""

    @abstractmethod
    def stream(self, messages: list[Message], params: GenerationParams) -> Iterator[str]:
        """답변을 조각(chunk) 단위로 흘려보낸다."""


class FakeBackend(ChatBackend):
    """테스트/화면 개발용. API 호출 없이 정해진 답을 돌려준다.

    replies 가 있으면 순서대로, 없으면 마지막 사용자 메시지를 따라 말한다.
    """

    def __init__(self, replies: list[str] | None = None, error: BackendError | None = None):
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[list[Message]] = []

    def _next(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self.error:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"echo: {last_user}"

    def generate(self, messages, params):
        return self._next(messages)

    def stream(self, messages, params):
        text = self._next(messages)
        for i in range(0, len(text), 4):
            yield text[i : i + 4]


def create_backend(entry: ModelEntry) -> ChatBackend:
    """설정의 backend 값에 맞는 백엔드를 만든다. 무거운 import 는 필요할 때만."""
    try:
        if entry.backend == "api":
            from chatbot.backends.api_backend import ApiBackend

            return ApiBackend(entry)
        if entry.backend == "local":
            from chatbot.backends.local_backend import LocalBackend

            return LocalBackend(entry)
        if entry.backend == "fake":
            return FakeBackend()
    except ConfigError as e:
        raise BackendError(str(e), status_code=500) from e
    raise BackendError(f"알 수 없는 backend 예요: {entry.backend}", status_code=500)
