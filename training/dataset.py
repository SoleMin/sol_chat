"""학습 데이터 로드: messages 형식 JSONL -> 대화 리스트.

한 줄에 대화 하나:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
from pathlib import Path

from chatbot.model import Message
from chatbot.preprocessing import normalize_messages


class DatasetError(ValueError):
    pass


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"데이터 파일이 없어요: {path}")
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno} JSON 형식 오류: {e}") from e
    return records


def load_conversations(path: str | Path, system_prompt: str = "") -> list[list[Message]]:
    """JSONL 을 읽어 서비스와 같은 형식(normalize_messages)의 대화 리스트로 만든다."""
    conversations = []
    for i, record in enumerate(load_jsonl(path), 1):
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list):
            raise DatasetError(f"{path} {i}번째 줄: 'messages' 리스트가 없어요.")
        try:
            conversations.append(normalize_messages(messages, system_prompt))
        except ValueError as e:
            raise DatasetError(f"{path} {i}번째 줄: {e}") from e
    if not conversations:
        raise DatasetError(f"{path} 에 대화가 없어요.")
    return conversations
