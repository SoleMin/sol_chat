"""API 요청/응답 데이터 형식."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, description="이어서 대화할 세션. 없으면 새 대화")
    message: str
    model: str | None = Field(None, description="models 키 이름. 없으면 세션 모델 또는 기본 모델")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionSummary(BaseModel):
    id: str
    title: str
    model: str
    updated_at: float
    message_count: int


class SessionDetail(SessionSummary):
    messages: list[ChatMessage]


class ModelInfo(BaseModel):
    name: str
    label: str
    is_default: bool


class ModelList(BaseModel):
    models: list[ModelInfo]
    default_model: str


class ErrorResponse(BaseModel):
    detail: str
