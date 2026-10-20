import pytest

from chatbot.config import ConfigError, load_chat_config, load_server_config


def test_load_repo_model_config():
    config = load_chat_config()
    assert config.default_model in config.models
    assert config.system_prompt
    entry = config.get_model(None)
    assert entry.backend == "api"
    assert entry.base_url.startswith("https://openrouter.ai")


def test_load_repo_server_config():
    config = load_server_config()
    assert config.rate_limit_per_minute > 0
    assert config.max_message_chars > 0


def test_unknown_model_raises():
    with pytest.raises(ConfigError):
        load_chat_config().get_model("없는-모델")


def test_missing_api_key_is_clear(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    entry = load_chat_config().get_model(None)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        entry.api_key()


def test_invalid_backend(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("models:\n  x:\n    backend: nope\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_chat_config(path)
