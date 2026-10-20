from fastapi.testclient import TestClient

from chatbot.config import ServerConfig
from chatbot.generation import ChatGenerator
from chatbot.model import FakeBackend
from server.main import create_app
from server.security import RateLimiter

CID = {"X-Client-Id": "client-aaaa-1111"}


def make_client(chat_config, **kwargs):
    gen = ChatGenerator(chat_config, backend_factory=lambda e: FakeBackend())
    return TestClient(create_app(chat_config, ServerConfig(**kwargs), gen))


def test_access_code_required(chat_config):
    client = make_client(chat_config, access_code="secret-code")
    assert client.get("/api/auth/check").status_code == 401
    assert client.get("/api/auth/check", headers={"X-Access-Code": "wrong"}).status_code == 401
    assert client.post("/api/chat", json={"message": "q"}, headers=CID).status_code == 401
    ok = {"X-Access-Code": "secret-code", **CID}
    assert client.get("/api/auth/check", headers=ok).status_code == 200
    assert client.post("/api/chat", json={"message": "q"}, headers=ok).status_code == 200


def test_chat_rate_limit_returns_429(chat_config):
    client = make_client(chat_config, access_code="c", rate_limit_per_minute=2)
    h = {"X-Access-Code": "c", **CID}
    codes = [client.post("/api/chat", json={"message": "q"}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    # 채팅 제한에 걸려도 대화 목록 조회는 된다
    assert client.get("/api/sessions", headers=h).status_code == 200


def test_api_rate_limit_returns_429(chat_config):
    client = make_client(chat_config, access_code="c", api_rate_limit_per_minute=3)
    h = {"X-Access-Code": "c"}
    codes = [client.get("/api/auth/check", headers=h).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_password_guessing_is_blocked(chat_config):
    client = make_client(chat_config, access_code="c", auth_fail_limit_per_minute=2)
    codes = [client.get("/api/auth/check", headers={"X-Access-Code": "guess"}).status_code for _ in range(3)]
    assert codes == [401, 401, 429]


def test_rate_limit_uses_cloudflare_ip(chat_config):
    client = make_client(chat_config, access_code="c", api_rate_limit_per_minute=1)
    h = {"X-Access-Code": "c"}
    assert client.get("/api/auth/check", headers={**h, "CF-Connecting-IP": "1.1.1.1"}).status_code == 200
    assert client.get("/api/auth/check", headers={**h, "CF-Connecting-IP": "2.2.2.2"}).status_code == 200
    assert client.get("/api/auth/check", headers={**h, "CF-Connecting-IP": "1.1.1.1"}).status_code == 429


def test_message_too_long_or_empty(chat_config):
    client = make_client(chat_config, access_code="c", max_message_chars=10)
    h = {"X-Access-Code": "c", **CID}
    assert client.post("/api/chat", json={"message": "가" * 11}, headers=h).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}, headers=h).status_code == 422


def test_rate_limiter_window():
    t = [0.0]
    limiter = RateLimiter(2, window_seconds=60, clock=lambda: t[0])
    assert limiter.hit("a") and limiter.hit("a") and not limiter.hit("a")
    t[0] = 60.0
    assert limiter.hit("a")


def test_no_code_allowed_when_not_required(chat_config):
    client = make_client(chat_config, access_code="", require_access_code=False)
    with client:
        assert client.get("/api/auth/check").status_code == 200
