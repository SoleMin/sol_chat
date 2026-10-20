"""CRITIC(20261020-baseline-CRITIC.md) B-001~B-006 회귀 테스트."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from chatbot.config import ServerConfig
from chatbot.conversation import ConversationStore
from chatbot.generation import ChatGenerator
from chatbot.model import BackendError, FakeBackend
from server.main import create_app
from server.security import RateLimiter


# B-001: 같은 모델의 첫 요청이 겹쳐도 백엔드는 한 번만 만든다
def test_backend_created_once_under_concurrency(chat_config):
    calls = []

    def factory(entry):
        calls.append(entry.name)
        time.sleep(0.2)
        return FakeBackend()

    gen = ChatGenerator(chat_config, backend_factory=factory)
    threads = [threading.Thread(target=gen.backend, args=("fake",)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["fake"]


def test_backend_init_failure_can_retry(chat_config):
    attempts = []

    def factory(entry):
        attempts.append(1)
        if len(attempts) == 1:
            raise BackendError("일시적 실패")
        return FakeBackend()

    gen = ChatGenerator(chat_config, backend_factory=factory)
    with pytest.raises(BackendError):
        gen.backend("fake")
    assert gen.backend("fake") is not None


def test_different_models_initialize_in_parallel(chat_config):
    def factory(entry):
        time.sleep(0.3)
        return FakeBackend()

    gen = ChatGenerator(chat_config, backend_factory=factory)
    start = time.monotonic()
    threads = [threading.Thread(target=gen.backend, args=(m,)) for m in ("fake", "fake2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - start < 0.55


# B-003: 세션 보관량 상한
def test_session_storage_is_capped():
    store = ConversationStore(max_messages_per_session=6)
    s = store.create("c1", "fake")
    for i in range(50):
        store.add_exchange(s, f"q{i}", f"a{i}")
    assert len(s.messages) == 6
    assert [m["role"] for m in s.messages] == ["user", "assistant"] * 3
    assert s.messages[-1]["content"] == "a49"


# B-005: 실패 한도를 넘으면 정답도 검증하지 않는다
def make_client(chat_config, backend=None, **kwargs):
    backend = backend or FakeBackend()
    gen = ChatGenerator(chat_config, backend_factory=lambda e: backend)
    return TestClient(create_app(chat_config, ServerConfig(**kwargs), gen))


def test_auth_lockout_blocks_correct_code_too(chat_config):
    client = make_client(chat_config, access_code="right", auth_fail_limit_per_minute=2)
    codes = [client.get("/api/auth/check", headers={"X-Access-Code": c}).status_code
             for c in ("w1", "w2", "w3", "right")]
    assert codes == [401, 401, 429, 429]


def test_rate_limiter_is_limited_and_cleanup():
    t = [0.0]
    limiter = RateLimiter(1, window_seconds=60, clock=lambda: t[0])
    assert not limiter.is_limited("a")
    limiter.hit("a")
    assert limiter.is_limited("a")
    for i in range(100):
        limiter.hit(f"ip{i}")
    t[0] = 1000.0
    limiter.cleanup()
    assert len(limiter) == 0


# B-006: 같은 대화에 동시 요청 -> 하나는 409
class SlowBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, messages, params):
        self.started.set()
        self.release.wait(5)
        return super().generate(messages, params)


def test_same_session_concurrent_request_conflict(chat_config):
    backend = SlowBackend()
    client = make_client(chat_config, backend, access_code="c")
    h = {"X-Access-Code": "c", "X-Client-Id": "client-aaaa-1111"}
    backend.release.set()
    sid = client.post("/api/chat", json={"message": "first"}, headers=h).json()["session_id"]
    backend.release.clear()
    backend.started.clear()

    results = {}
    t = threading.Thread(target=lambda: results.setdefault(
        "a", client.post("/api/chat", json={"session_id": sid, "message": "A"}, headers=h).status_code))
    t.start()
    assert backend.started.wait(5)
    second = client.post("/api/chat", json={"session_id": sid, "message": "B"}, headers=h)
    backend.release.set()
    t.join()
    assert results["a"] == 200 and second.status_code == 409
    # 다른 대화는 막히지 않음은 test_api 의 일반 경로로 확인


def test_session_unlocked_after_stream_error(chat_config):
    client = make_client(chat_config, FakeBackend(error=BackendError("x")), access_code="c")
    h = {"X-Access-Code": "c", "X-Client-Id": "client-aaaa-1111"}
    r = client.post("/api/chat/stream", json={"message": "q"}, headers=h)
    sid = r.text.split('"session_id": "')[1].split('"')[0]
    r2 = client.post("/api/chat/stream", json={"session_id": sid, "message": "q"}, headers=h)
    assert r2.status_code == 200


# B-002: 로컬 스트리밍 생성 스레드 예외 전달
def test_local_stream_propagates_thread_error():
    pytest.importorskip("transformers")
    from chatbot.backends.local_backend import LocalBackend
    from chatbot.config import GenerationParams, ModelEntry

    class Tok:
        pad_token_id = 0

        def apply_chat_template(self, *a, **k):
            return {}

        def decode(self, *a, **k):
            return ""

    class BoomModel:
        device = "cpu"

        def generate(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    entry = ModelEntry(name="l", label="l", backend="local", base_model="x")
    backend = LocalBackend(entry, model=BoomModel(), tokenizer=Tok())
    done = []

    def consume():
        try:
            list(backend.stream([], GenerationParams()))
        except BackendError as e:
            done.append(e.user_message)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    t.join(5)
    assert done and "생성" in done[0]
