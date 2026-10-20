"""설정 로드: configs/*.yaml + .env -> dataclass.

다른 모듈은 yaml 을 직접 읽지 않고 이 파일의 dataclass 만 사용한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_SERVER_CONFIG = PROJECT_ROOT / "configs" / "server.yaml"


class ConfigError(Exception):
    """설정 파일이나 환경 변수가 잘못되었을 때."""


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024


@dataclass
class HistoryConfig:
    max_turns: int = 10
    max_chars: int = 12000


@dataclass
class ModelEntry:
    """configs/model.yaml 의 models 항목 하나."""

    name: str
    label: str
    backend: str  # "api" | "local" | "fake"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    base_model: str = ""
    adapter_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str:
        """api_key_env 에 적힌 환경 변수에서 키를 읽는다. 없으면 ConfigError."""
        if not self.api_key_env:
            raise ConfigError(f"모델 '{self.name}'에 api_key_env 가 설정되지 않았어요.")
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"환경 변수 {self.api_key_env} 가 비어 있어요. .env 파일에 API 키를 넣어주세요."
            )
        return key


@dataclass
class ChatConfig:
    system_prompt: str
    default_model: str
    models: dict[str, ModelEntry]
    generation: GenerationParams = field(default_factory=GenerationParams)
    history: HistoryConfig = field(default_factory=HistoryConfig)

    def get_model(self, name: str | None) -> ModelEntry:
        key = name or self.default_model
        if key not in self.models:
            raise ConfigError(f"등록되지 않은 모델이에요: {key}")
        return self.models[key]


@dataclass
class SessionLimits:
    max_per_client: int = 30
    max_total: int = 500
    ttl_minutes: int = 1440
    max_turns_per_session: int = 100


@dataclass
class ServerConfig:
    access_code: str = ""
    require_access_code: bool = True
    rate_limit_per_minute: int = 30
    api_rate_limit_per_minute: int = 120
    auth_fail_limit_per_minute: int = 10
    max_message_chars: int = 2000
    sessions: SessionLimits = field(default_factory=SessionLimits)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"설정 파일이 없어요: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"설정 파일 형식이 잘못됐어요: {path}")
    return data


def load_env() -> None:
    """프로젝트 루트의 .env 를 환경 변수로 읽는다 (이미 있는 값은 덮어쓰지 않음)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_chat_config(path: str | Path | None = None) -> ChatConfig:
    load_env()
    data = _read_yaml(Path(path) if path else DEFAULT_MODEL_CONFIG)

    raw_models = data.get("models") or {}
    if not raw_models:
        raise ConfigError("models 에 모델이 하나 이상 있어야 해요.")
    known = {"label", "backend", "model", "base_url", "api_key_env", "base_model", "adapter_path"}
    models = {}
    for name, raw in raw_models.items():
        raw = raw or {}
        if raw.get("backend") not in ("api", "local", "fake"):
            raise ConfigError(f"모델 '{name}'의 backend 는 api / local / fake 중 하나여야 해요.")
        models[name] = ModelEntry(
            name=name,
            label=raw.get("label", name),
            backend=raw["backend"],
            model=raw.get("model", ""),
            base_url=raw.get("base_url", ""),
            api_key_env=raw.get("api_key_env", ""),
            base_model=raw.get("base_model", ""),
            adapter_path=raw.get("adapter_path", ""),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    default_model = data.get("default_model") or next(iter(models))
    if default_model not in models:
        raise ConfigError(f"default_model '{default_model}' 이 models 에 없어요.")

    return ChatConfig(
        system_prompt=(data.get("system_prompt") or "").strip(),
        default_model=default_model,
        models=models,
        generation=GenerationParams(**(data.get("generation") or {})),
        history=HistoryConfig(**(data.get("history") or {})),
    )


def load_server_config(path: str | Path | None = None) -> ServerConfig:
    load_env()
    data = _read_yaml(Path(path) if path else DEFAULT_SERVER_CONFIG)
    return ServerConfig(
        access_code=os.environ.get("ACCESS_CODE", "").strip(),
        require_access_code=bool(data.get("require_access_code", True)),
        rate_limit_per_minute=int(data.get("rate_limit_per_minute", 30)),
        api_rate_limit_per_minute=int(data.get("api_rate_limit_per_minute", 120)),
        auth_fail_limit_per_minute=int(data.get("auth_fail_limit_per_minute", 10)),
        max_message_chars=int(data.get("max_message_chars", 2000)),
        sessions=SessionLimits(**(data.get("sessions") or {})),
    )
