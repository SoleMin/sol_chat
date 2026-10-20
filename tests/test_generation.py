import pytest

from chatbot.generation import ChatGenerator
from chatbot.model import BackendError, FakeBackend


def make(chat_config, backend):
    return ChatGenerator(chat_config, backend_factory=lambda entry: backend)


def test_reply_builds_messages_and_cleans(chat_config):
    backend = FakeBackend(replies=["  답변입니다<|im_end|> "])
    gen = make(chat_config, backend)
    history = [{"role": "user", "content": "이전 질문"}, {"role": "assistant", "content": "이전 답"}]
    assert gen.reply(None, history, "새 질문") == "답변입니다"
    sent = backend.calls[0]
    assert sent[0] == {"role": "system", "content": "너는 테스트 봇이야."}
    assert sent[1]["content"] == "이전 질문"
    assert sent[-1] == {"role": "user", "content": "새 질문"}


def test_stream_reply_joins_to_full_text(chat_config):
    gen = make(chat_config, FakeBackend(replies=["스트리밍 답변이에요"]))
    assert "".join(gen.stream_reply(None, [], "q")) == "스트리밍 답변이에요"


def test_unknown_model_is_backend_error(chat_config):
    gen = make(chat_config, FakeBackend())
    with pytest.raises(BackendError) as e:
        gen.reply("없는모델", [], "q")
    assert e.value.status_code == 400


def test_backend_is_cached_per_model(chat_config):
    created = []

    def factory(entry):
        created.append(entry.name)
        return FakeBackend()

    gen = ChatGenerator(chat_config, backend_factory=factory)
    gen.reply("fake", [], "q")
    gen.reply("fake", [], "q")
    gen.reply("fake2", [], "q")
    assert created == ["fake", "fake2"]


def test_history_limit_applied(chat_config):
    backend = FakeBackend()
    gen = make(chat_config, backend)
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    gen.reply(None, history, "new")
    # max_turns=3: 시스템 + 최근 3턴(6개) + 새 질문
    assert len(backend.calls[0]) == 1 + 6 + 1
