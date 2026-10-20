"""입력 정리와 messages 변환.

서비스(대화)와 학습(데이터)이 같은 함수로 messages 를 만들어서,
학습 때와 서비스 때 모델이 보는 형식이 어긋나지 않게 한다.
"""

from __future__ import annotations

import re
import unicodedata

from chatbot.model import Message

ROLES = ("system", "user", "assistant")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """유니코드 정규화, 제어 문자 제거, 줄바꿈 통일, 앞뒤 공백 제거."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    return text.strip()


def validate_user_input(text: str, max_chars: int | None = None) -> str:
    """사용자 입력을 정리하고, 비었거나 너무 길면 ValueError."""
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("메시지가 비어 있어요.")
    if max_chars is not None and len(cleaned) > max_chars:
        raise ValueError(f"메시지가 너무 길어요. {max_chars}자 이하로 보내주세요.")
    return cleaned


def trim_history(history: list[Message], max_turns: int, max_chars: int) -> list[Message]:
    """오래된 메시지부터 잘라 최근 max_turns 턴, max_chars 글자 이내로 만든다.

    1턴 = user 메시지 1개 + 그 답변. 잘린 결과는 항상 user 메시지로 시작한다.
    """
    history = [m for m in history if m["role"] in ("user", "assistant")]

    # 최근 max_turns 개의 user 메시지부터 시작
    user_idx = [i for i, m in enumerate(history) if m["role"] == "user"]
    if max_turns <= 0 or not user_idx:
        return []
    start = user_idx[-max_turns] if len(user_idx) >= max_turns else user_idx[0]
    trimmed = history[start:]

    # 글자 수 초과 시 앞에서부터 턴 단위로 제거
    while trimmed and sum(len(m["content"]) for m in trimmed) > max_chars:
        next_user = next((i for i, m in enumerate(trimmed) if i > 0 and m["role"] == "user"), None)
        trimmed = trimmed[next_user:] if next_user else []
    return trimmed


def build_messages(
    system_prompt: str,
    history: list[Message],
    user_message: str,
    *,
    max_turns: int,
    max_chars: int,
) -> list[Message]:
    """시스템 프롬프트 + (잘린) 히스토리 + 이번 사용자 메시지 -> 모델 입력 messages."""
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(trim_history(history, max_turns, max_chars))
    messages.append({"role": "user", "content": user_message})
    return messages


def normalize_messages(messages: list[Message], system_prompt: str = "") -> list[Message]:
    """학습 데이터의 대화 한 개를 서비스와 같은 형식으로 맞춘다.

    - 내용 정리(clean_text), 빈 메시지 제거
    - system 메시지가 없고 system_prompt 가 있으면 맨 앞에 추가
    - 잘못된 role 이나 user/assistant 순서가 어긋나면 ValueError
    """
    out: list[Message] = []
    for m in messages:
        role = m.get("role")
        if role not in ROLES:
            raise ValueError(f"알 수 없는 role: {role}")
        content = clean_text(m.get("content", ""))
        if content:
            out.append({"role": role, "content": content})

    if out and out[0]["role"] == "system":
        system, rest = out[:1], out[1:]
    else:
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        rest = out

    if not rest or rest[0]["role"] != "user" or rest[-1]["role"] != "assistant":
        raise ValueError("대화는 user 로 시작하고 assistant 로 끝나야 해요.")
    for prev, cur in zip(rest, rest[1:]):
        if prev["role"] == cur["role"] or cur["role"] == "system":
            raise ValueError("user 와 assistant 가 번갈아 나와야 해요.")
    return system + rest
