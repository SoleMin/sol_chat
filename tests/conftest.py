import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.config import ChatConfig, GenerationParams, HistoryConfig, ModelEntry  # noqa: E402


@pytest.fixture
def chat_config() -> ChatConfig:
    return ChatConfig(
        system_prompt="너는 테스트 봇이야.",
        default_model="fake",
        models={
            "fake": ModelEntry(name="fake", label="Fake", backend="fake"),
            "fake2": ModelEntry(name="fake2", label="Fake 2", backend="fake"),
        },
        generation=GenerationParams(),
        history=HistoryConfig(max_turns=3, max_chars=1000),
    )
