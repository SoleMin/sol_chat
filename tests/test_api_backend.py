"""ApiBackend 를 실제 네트워크 없이 가짜 OpenAI 클라이언트로 확인."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from chatbot.backends.api_backend import ApiBackend, to_backend_error
from chatbot.config import GenerationParams, ModelEntry
from chatbot.model import BackendError

ENTRY = ModelEntry(name="m", label="M", backend="api", model="google/gemini-2.5-flash", base_url="https://x/v1")


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_generate_passes_params():
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="안녕"))])
    comp = FakeCompletions(response=resp)
    backend = ApiBackend(ENTRY, client=client(comp))
    assert backend.generate([{"role": "user", "content": "hi"}], GenerationParams(temperature=0.1)) == "안녕"
    assert comp.kwargs["model"] == "google/gemini-2.5-flash"
    assert comp.kwargs["temperature"] == 0.1
    assert comp.kwargs["stream"] is False


def test_stream_yields_deltas():
    def chunk(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    comp = FakeCompletions(response=iter([chunk("안"), chunk(None), SimpleNamespace(choices=[]), chunk("녕")]))
    backend = ApiBackend(ENTRY, client=client(comp))
    assert list(backend.stream([], GenerationParams())) == ["안", "녕"]


def _status_error(cls, code):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return cls("err", response=httpx.Response(code, request=req), body=None)


def test_auth_error_is_mapped():
    comp = FakeCompletions(error=_status_error(openai.AuthenticationError, 401))
    with pytest.raises(BackendError, match="API 키"):
        ApiBackend(ENTRY, client=client(comp)).generate([], GenerationParams())


def test_credit_error_is_mapped():
    err = to_backend_error(_status_error(openai.APIStatusError, 402))
    assert "크레딧" in err.user_message


def test_missing_key_raises_backend_error(monkeypatch):
    from chatbot.model import create_backend

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    entry = ModelEntry(**{**ENTRY.__dict__, "api_key_env": "OPENROUTER_API_KEY"})
    with pytest.raises(BackendError, match="OPENROUTER_API_KEY"):
        create_backend(entry)
