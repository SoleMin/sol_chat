"""모델 출력 정리."""

from __future__ import annotations

import re

# 로컬 모델이 가끔 그대로 내보내는 특수 토큰들
_SPECIAL_TOKENS = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|<\|eot_id\|>|</s>|<s>")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_chunk(chunk: str) -> str:
    """스트리밍 조각용: 특수 토큰만 제거 (공백은 조각 사이 이음새라 건드리지 않음)."""
    return _SPECIAL_TOKENS.sub("", chunk)


def clean_response(text: str) -> str:
    """완성된 답변용: 특수 토큰 제거, 과한 빈 줄 축소, 앞뒤 공백 제거."""
    text = clean_chunk(text or "")
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
