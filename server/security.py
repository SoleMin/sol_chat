"""외부 공개용 보호장치: 접속 비밀번호, 요청 수 제한, 클라이언트 식별."""

from __future__ import annotations

import hmac
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import HTTPException, Request

ACCESS_HEADER = "X-Access-Code"
CLIENT_HEADER = "X-Client-Id"
_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class RateLimiter:
    """키(IP)별로 최근 window 초 동안의 요청 수를 센다 (슬라이딩 윈도우)."""

    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = clock()

    def _prune(self, hits: deque[float], now: float) -> None:
        while hits and now - hits[0] >= self.window:
            hits.popleft()

    def hit(self, key: str) -> bool:
        """요청을 기록하고, 허용되면 True."""
        now = self.clock()
        with self._lock:
            if now - self._last_cleanup >= self.window:
                self._cleanup_locked(now)
            hits = self._hits[key]
            self._prune(hits, now)
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def is_limited(self, key: str) -> bool:
        """기록하지 않고 지금 한도에 걸려 있는지만 본다."""
        now = self.clock()
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return False
            self._prune(hits, now)
            return len(hits) >= self.limit

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_locked(self.clock())

    def _cleanup_locked(self, now: float) -> None:
        # 더 이상 요청하지 않는 IP 의 기록을 지워 메모리가 계속 늘지 않게 한다
        for key in list(self._hits):
            self._prune(self._hits[key], now)
            if not self._hits[key]:
                del self._hits[key]
        self._last_cleanup = now

    def __len__(self) -> int:
        return len(self._hits)


def client_ip(request: Request) -> str:
    # Cloudflare Tunnel 을 거치면 서버 입장에서는 모두 localhost 이므로 CF 헤더를 우선 사용
    return (
        request.headers.get("CF-Connecting-IP")
        or (request.client.host if request.client else "")
        or "unknown"
    )


def client_id(request: Request) -> str:
    """브라우저가 만든 클라이언트 ID. 대화 목록을 사람별로 나누는 기준."""
    value = request.headers.get(CLIENT_HEADER, "")
    if not _CLIENT_ID.match(value):
        raise HTTPException(status_code=400, detail=f"{CLIENT_HEADER} 헤더가 필요해요.")
    return value


class Limiters:
    """서버에서 쓰는 요청 제한 세 가지."""

    def __init__(self, *, chat: int, api: int, auth_fail: int, clock: Callable[[], float] = time.monotonic):
        self.chat = RateLimiter(chat, clock=clock)  # 채팅 메시지 (크레딧이 드는 요청)
        self.api = RateLimiter(api, clock=clock)  # 모든 /api 요청
        self.auth_fail = RateLimiter(auth_fail, clock=clock)  # 비밀번호 실패


_TOO_MANY = "요청이 너무 많아요. 1분 정도 기다렸다가 다시 시도해주세요."
_LOCKED = "비밀번호를 너무 많이 틀렸어요. 1분 뒤에 다시 시도해주세요."


def check_api_rate_limit(request: Request) -> None:
    if not request.app.state.limiters.api.hit(client_ip(request)):
        raise HTTPException(status_code=429, detail=_TOO_MANY)


def check_chat_rate_limit(request: Request) -> None:
    if not request.app.state.limiters.chat.hit(client_ip(request)):
        raise HTTPException(status_code=429, detail=_TOO_MANY)


def check_access_code(request: Request) -> None:
    expected: str = request.app.state.server_config.access_code
    if not expected:
        return  # require_access_code: false 이고 비밀번호를 안 정한 경우 (로컬 개발용)
    # 틀린 비밀번호는 IP별로 세서, 너무 많이 틀리면 잠시 막는다 (비밀번호 추측 방지).
    # 막힌 동안에는 비밀번호를 비교하지도 않는다 (맞는 값을 골라낼 수 없게).
    fails: RateLimiter = request.app.state.limiters.auth_fail
    ip = client_ip(request)
    if fails.is_limited(ip):
        raise HTTPException(status_code=429, detail=_LOCKED)
    given = request.headers.get(ACCESS_HEADER, "")
    if hmac.compare_digest(given.encode(), expected.encode()):
        return
    if not fails.hit(ip):
        raise HTTPException(status_code=429, detail=_LOCKED)
    raise HTTPException(status_code=401, detail="접속 비밀번호가 올바르지 않아요.")
