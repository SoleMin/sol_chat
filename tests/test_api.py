import json

import pytest
from fastapi.testclient import TestClient

from chatbot.config import ServerConfig
from chatbot.generation import ChatGenerator
from chatbot.model import BackendError, FakeBackend
from server.main import create_app

CODE = "test-access-code"
H1 = {"X-Access-Code": CODE, "X-Client-Id": "client-aaaa-1111"}
H2 = {"X-Access-Code": CODE, "X-Client-Id": "client-bbbb-2222"}


def make_client(chat_config, backend=None, **server_kwargs):
    backend = backend or FakeBackend()
    server_config = ServerConfig(access_code=CODE, **server_kwargs)
    gen = ChatGenerator(chat_config, backend_factory=lambda entry: backend)
    return TestClient(create_app(chat_config, server_config, gen)), backend


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_returns_json_and_session(chat_config):
    client, _ = make_client(chat_config, FakeBackend(replies=["안녕하세요!"]))
    r = client.post("/api/chat", json={"session_id": None, "message": "안녕"}, headers=H1)
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "안녕하세요!" and body["session_id"] and body["model"] == "fake"


def test_multi_turn_history_is_sent(chat_config):
    client, backend = make_client(chat_config)
    sid = client.post("/api/chat", json={"message": "첫 질문"}, headers=H1).json()["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "두번째"}, headers=H1)
    contents = [m["content"] for m in backend.calls[1]]
    assert "첫 질문" in contents and "echo: 첫 질문" in contents and contents[-1] == "두번째"


def test_stream_events(chat_config):
    client, _ = make_client(chat_config, FakeBackend(replies=["스트리밍 테스트 답변"]))
    r = client.post("/api/chat/stream", json={"message": "q"}, headers=H1)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0][0] == "session" and events[-1][0] == "done"
    assert "".join(d["text"] for e, d in events if e == "token") == "스트리밍 테스트 답변"
    sid = events[0][1]["session_id"]
    detail = client.get(f"/api/sessions/{sid}", headers=H1).json()
    assert detail["messages"][-1]["content"] == "스트리밍 테스트 답변"


def test_stream_error_does_not_save(chat_config):
    client, _ = make_client(chat_config, FakeBackend(error=BackendError("API 키가 올바르지 않아요.")))
    events = parse_sse(client.post("/api/chat/stream", json={"message": "q"}, headers=H1).text)
    assert events[-1] == ("error", {"detail": "API 키가 올바르지 않아요."})
    sid = events[0][1]["session_id"]
    assert client.get(f"/api/sessions/{sid}", headers=H1).json()["message_count"] == 0


def test_backend_error_is_readable_json(chat_config):
    client, _ = make_client(chat_config, FakeBackend(error=BackendError("API 키가 올바르지 않아요.")))
    r = client.post("/api/chat", json={"message": "q"}, headers=H1)
    assert r.status_code == 502 and r.json()["detail"] == "API 키가 올바르지 않아요."


def test_sessions_are_isolated_per_client(chat_config):
    client, _ = make_client(chat_config)
    sid = client.post("/api/chat", json={"message": "비밀 대화"}, headers=H1).json()["session_id"]
    assert len(client.get("/api/sessions", headers=H1).json()) == 1
    assert client.get("/api/sessions", headers=H2).json() == []
    assert client.get(f"/api/sessions/{sid}", headers=H2).status_code == 404
    assert client.post("/api/chat", json={"session_id": sid, "message": "x"}, headers=H2).status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=H2).status_code == 404
    assert client.delete(f"/api/sessions/{sid}", headers=H1).status_code == 200


def test_models_endpoint_and_model_switch(chat_config):
    client, _ = make_client(chat_config)
    models = client.get("/api/models", headers=H1).json()
    assert models["default_model"] == "fake" and [m["name"] for m in models["models"]] == ["fake", "fake2"]
    r = client.post("/api/chat", json={"message": "q", "model": "fake2"}, headers=H1)
    assert r.json()["model"] == "fake2"
    assert client.post("/api/chat", json={"message": "q", "model": "nope"}, headers=H1).status_code == 400


def test_client_id_required(chat_config):
    client, _ = make_client(chat_config)
    r = client.post("/api/chat", json={"message": "q"}, headers={"X-Access-Code": CODE})
    assert r.status_code == 400


def test_web_page_served(chat_config):
    client, _ = make_client(chat_config)
    r = client.get("/")
    assert r.status_code == 200 and "sol_chat" in r.text


def test_startup_requires_access_code(chat_config):
    app = create_app(chat_config, ServerConfig(access_code="", require_access_code=True), ChatGenerator(chat_config))
    with pytest.raises(RuntimeError, match="ACCESS_CODE"):
        with TestClient(app):
            pass
