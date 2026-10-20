"""채팅 API 엔드포인트.

모든 /api 요청은 전체 요청 수 제한 -> 접속 비밀번호 검사를 거치고,
채팅 요청은 채팅 메시지 수 제한을 한 번 더 거친다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from chatbot.conversation import ConversationStore, Session, SessionBusy, SessionNotFound
from chatbot.generation import ChatGenerator
from chatbot.model import BackendError
from chatbot.postprocessing import clean_response
from chatbot.preprocessing import validate_user_input
from server.schemas import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelList,
    SessionDetail,
    SessionSummary,
)
from server.security import check_access_code, check_api_rate_limit, check_chat_rate_limit, client_id

router = APIRouter(prefix="/api", dependencies=[Depends(check_api_rate_limit), Depends(check_access_code)])


def _store(request: Request) -> ConversationStore:
    return request.app.state.store


def _generator(request: Request) -> ChatGenerator:
    return request.app.state.generator


def _summary(s: Session) -> SessionSummary:
    return SessionSummary(id=s.id, title=s.title, model=s.model, updated_at=s.updated_at, message_count=len(s.messages))


def _prepare(request: Request, body: ChatRequest, cid: str) -> tuple[Session, str, str]:
    """요청 검증 후 (세션, 정리된 메시지, 사용할 모델 이름).

    성공하면 세션이 "생성 중"으로 잠긴다. 호출한 쪽이 끝나면 반드시 store.release().
    """
    try:
        message = validate_user_input(body.message, request.app.state.server_config.max_message_chars)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config = _generator(request).config
    model = body.model
    if model is not None and model not in config.models:
        raise HTTPException(status_code=400, detail=f"등록되지 않은 모델이에요: {model}")

    store = _store(request)
    if body.session_id:
        try:
            session = store.get(cid, body.session_id)
        except SessionNotFound as e:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없어요. 새 대화를 시작해주세요.") from e
    else:
        session = store.create(cid, model or config.default_model)
    try:
        store.acquire(session)
    except SessionBusy as e:
        raise HTTPException(status_code=409, detail="이 대화에서 이미 답변을 만들고 있어요. 끝난 뒤에 보내주세요.") from e
    if model:
        session.model = model
    return session, message, session.model


@router.get("/auth/check")
def auth_check() -> dict:
    return {"ok": True}


@router.get("/models", response_model=ModelList)
def list_models(request: Request) -> ModelList:
    config = _generator(request).config
    return ModelList(
        models=[ModelInfo(name=n, label=e.label, is_default=n == config.default_model) for n, e in config.models.items()],
        default_model=config.default_model,
    )


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(request: Request, cid: str = Depends(client_id)) -> list[SessionSummary]:
    return [_summary(s) for s in _store(request).list(cid)]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, request: Request, cid: str = Depends(client_id)) -> SessionDetail:
    try:
        s = _store(request).get(cid, session_id)
    except SessionNotFound as e:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없어요.") from e
    return SessionDetail(**_summary(s).model_dump(), messages=s.messages)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request, cid: str = Depends(client_id)) -> dict:
    try:
        _store(request).delete(cid, session_id)
    except SessionNotFound as e:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없어요.") from e
    return {"ok": True}


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(check_chat_rate_limit)])
def chat(body: ChatRequest, request: Request, cid: str = Depends(client_id)) -> ChatResponse:
    session, message, model = _prepare(request, body, cid)
    store = _store(request)
    try:
        reply = _generator(request).reply(model, session.messages, message)  # BackendError -> main.py 핸들러
        store.add_exchange(session, message, reply)
    finally:
        store.release(session)
    return ChatResponse(session_id=session.id, reply=reply, model=model)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", dependencies=[Depends(check_chat_rate_limit)])
def chat_stream(body: ChatRequest, request: Request, cid: str = Depends(client_id)) -> StreamingResponse:
    """SSE 스트림. 이벤트: session -> token(여러 번) -> done, 실패 시 error."""
    session, message, model = _prepare(request, body, cid)
    generator, store = _generator(request), _store(request)

    def events() -> Iterator[str]:
        try:
            yield _sse("session", {"session_id": session.id, "model": model})
            parts: list[str] = []
            try:
                for chunk in generator.stream_reply(model, session.messages, message):
                    parts.append(chunk)
                    yield _sse("token", {"text": chunk})
            except BackendError as e:
                yield _sse("error", {"detail": e.user_message})
                return
            reply = clean_response("".join(parts))
            store.add_exchange(session, message, reply)
            yield _sse("done", {"session_id": session.id, "title": session.title})
        finally:
            # 정상 종료, 오류, 연결 끊김 모두에서 잠금 해제
            # (스트림이 시작도 못 하고 끊긴 경우는 conversation 의 busy_timeout 이 처리)
            store.release(session)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
