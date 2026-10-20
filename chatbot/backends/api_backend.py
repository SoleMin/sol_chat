"""OpenAI 호환 chat completions API 백엔드.

OpenRouter 를 기본으로 쓰지만, base_url 만 바꾸면 Colab 에 띄운 vLLM 서버 등
OpenAI 호환 서버라면 어디든 같은 코드로 호출할 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterator

import openai

from chatbot.config import GenerationParams, ModelEntry
from chatbot.model import BackendError, ChatBackend, Message

# OpenRouter 가 요청 출처를 표시할 때 쓰는 선택 헤더 (다른 서버는 무시함)
_OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/SoleMin/sol_chat", "X-Title": "sol_chat"}


class ApiBackend(ChatBackend):
    def __init__(self, entry: ModelEntry, client: openai.OpenAI | None = None):
        if not entry.model:
            raise BackendError(f"모델 '{entry.name}'에 model(모델 ID)이 없어요.", status_code=500)
        self.entry = entry
        self.client = client or openai.OpenAI(
            api_key=entry.api_key(),
            base_url=entry.base_url or None,
            default_headers=_OPENROUTER_HEADERS,
            timeout=60,
            max_retries=1,
        )

    def _request(self, messages: list[Message], params: GenerationParams, stream: bool):
        return self.client.chat.completions.create(
            model=self.entry.model,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stream=stream,
        )

    def generate(self, messages, params):
        try:
            response = self._request(messages, params, stream=False)
        except openai.OpenAIError as e:
            raise to_backend_error(e) from e
        if not response.choices:
            raise BackendError("모델이 빈 응답을 보냈어요. 잠시 후 다시 시도해주세요.")
        return response.choices[0].message.content or ""

    def stream(self, messages, params) -> Iterator[str]:
        try:
            for chunk in self._request(messages, params, stream=True):
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    yield text
        except openai.OpenAIError as e:
            raise to_backend_error(e) from e


def to_backend_error(e: openai.OpenAIError) -> BackendError:
    """API 오류를 사용자에게 보여줄 수 있는 메시지로 바꾼다."""
    if isinstance(e, openai.AuthenticationError):
        return BackendError("API 키가 올바르지 않아요. .env 의 OPENROUTER_API_KEY 를 확인해주세요.")
    if isinstance(e, openai.RateLimitError):
        return BackendError("API 호출 한도를 넘었어요. 잠시 후 다시 시도해주세요.", status_code=503)
    if isinstance(e, openai.APITimeoutError):
        return BackendError("모델 응답이 너무 오래 걸려요. 다시 시도해주세요.", status_code=504)
    if isinstance(e, openai.APIConnectionError):
        return BackendError("모델 서버에 연결하지 못했어요. 인터넷 연결이나 base_url 을 확인해주세요.")
    if isinstance(e, openai.NotFoundError):
        return BackendError("모델을 찾을 수 없어요. configs/model.yaml 의 model ID 를 확인해주세요.")
    if isinstance(e, openai.APIStatusError):
        if e.status_code == 402:
            return BackendError("OpenRouter 크레딧이 부족해요. 크레딧을 충전하거나 무료 모델을 사용해주세요.")
        return BackendError(f"모델 API 오류가 났어요 (HTTP {e.status_code}). 잠시 후 다시 시도해주세요.")
    return BackendError("모델 호출 중 알 수 없는 오류가 났어요.")
