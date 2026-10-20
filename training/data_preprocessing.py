"""학습용 전처리: 대화 -> 토큰 + 라벨.

핵심: 모델이 "답변(assistant) 부분"만 배우도록, 나머지 토큰(시스템 프롬프트, 사용자 질문,
chat template 의 특수 토큰)의 라벨은 -100 으로 가린다 (loss 계산에서 제외).
"""

from __future__ import annotations

import random

from chatbot.model import Message

IGNORE_INDEX = -100


def split_train_valid(items: list, valid_ratio: float, seed: int = 42) -> tuple[list, list]:
    """섞어서 검증용을 떼어낸다. 데이터가 적어도 학습용은 최소 1개 남긴다."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n_valid = int(len(items) * valid_ratio)
    if valid_ratio > 0 and n_valid == 0 and len(items) >= 2:
        n_valid = 1
    return items[n_valid:], items[:n_valid]


def _ids(tokenizer, messages: list[Message], add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True
    )
    return list(out["input_ids"])


def tokenize_conversation(messages: list[Message], tokenizer, max_length: int) -> dict[str, list[int]]:
    """대화 하나를 input_ids / attention_mask / labels 로 만든다.

    assistant 메시지마다 "그 앞까지 + 답변 시작 프롬프트" 와 "답변까지 포함" 을 각각 토크나이즈해서,
    늘어난 구간을 답변 토큰으로 본다. (chat template 이 앞부분을 바꾸지 않는 모델 기준: Qwen, Llama 등)
    """
    input_ids = _ids(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(input_ids)

    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(_ids(tokenizer, messages[:i], add_generation_prompt=True))
        end = len(_ids(tokenizer, messages[: i + 1], add_generation_prompt=False))
        for j in range(start, min(end, len(labels))):
            labels[j] = input_ids[j]

    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def tokenize_all(conversations: list[list[Message]], tokenizer, max_length: int) -> list[dict[str, list[int]]]:
    """전부 토크나이즈하고, 잘려서 배울 답변이 하나도 안 남은 대화는 버린다."""
    features = [tokenize_conversation(c, tokenizer, max_length) for c in conversations]
    return [f for f in features if any(label != IGNORE_INDEX for label in f["labels"])]


def pad_batch(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """배치 안에서 가장 긴 길이에 맞춰 오른쪽을 채운다 (라벨은 -100 으로)."""
    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
    return batch
