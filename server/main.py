"""FastAPI 앱 진입점.

    uvicorn server.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from chatbot.config import ChatConfig, ServerConfig, load_chat_config, load_server_config
from chatbot.conversation import ConversationStore
from chatbot.generation import ChatGenerator
from chatbot.model import BackendError
from server.routes import router
from server.security import Limiters

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(
    chat_config: ChatConfig | None = None,
    server_config: ServerConfig | None = None,
    generator: ChatGenerator | None = None,
) -> FastAPI:
    """테스트에서는 설정/생성기를 주입하고, 실행할 때는 configs/ 와 .env 를 읽는다."""
    chat_config = chat_config or load_chat_config()
    server_config = server_config or load_server_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if server_config.require_access_code and not server_config.access_code:
            raise RuntimeError(
                ".env 에 ACCESS_CODE(접속 비밀번호)를 설정해주세요. "
                "로컬에서만 쓸 거라면 configs/server.yaml 의 require_access_code 를 false 로 바꿀 수 있어요."
            )
        yield

    app = FastAPI(title="sol_chat", lifespan=lifespan)
    app.state.server_config = server_config
    app.state.generator = generator or ChatGenerator(chat_config)
    app.state.store = ConversationStore(
        max_per_client=server_config.sessions.max_per_client,
        max_total=server_config.sessions.max_total,
        ttl_seconds=server_config.sessions.ttl_minutes * 60,
        max_messages_per_session=server_config.sessions.max_turns_per_session * 2,
    )
    app.state.limiters = Limiters(
        chat=server_config.rate_limit_per_minute,
        api=server_config.api_rate_limit_per_minute,
        auth_fail=server_config.auth_fail_limit_per_minute,
    )

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.user_message})

    app.include_router(router)
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
