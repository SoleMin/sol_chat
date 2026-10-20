"""학습 데이터 로드/전처리 (torch, transformers 없이)."""

import json

import pytest

from chatbot.config import PROJECT_ROOT

from training.data_preprocessing import IGNORE_INDEX, pad_batch, split_train_valid, tokenize_all, tokenize_conversation
from training.dataset import DatasetError, load_conversations


class CharTokenizer:
    """글자 하나 = 토큰 하나인 가짜 토크나이저. Qwen 과 비슷한 chat template."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        ids = [ord(c) for c in text]
        return {"input_ids": ids} if return_dict else ids


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_sample_data_is_valid():
    convs = load_conversations(PROJECT_ROOT / "data/examples/sample.jsonl", system_prompt="sys")
    assert len(convs) >= 30
    assert all(c[0] == {"role": "system", "content": "sys"} for c in convs)
    assert all(c[-1]["role"] == "assistant" for c in convs)


def test_bad_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        + "\n" + json.dumps({"messages": [{"role": "assistant", "content": "a"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="2번째"):
        load_conversations(path)


def test_only_assistant_tokens_are_labeled():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    f = tokenize_conversation(msgs, CharTokenizer(), max_length=1000)
    learned = decode([t for t, label in zip(f["input_ids"], f["labels"]) if label != IGNORE_INDEX])
    assert learned == "A1</>A2</>"  # 답변 + 끝 표시만 학습, 질문/시스템은 제외
    assert len(f["input_ids"]) == len(f["labels"]) == len(f["attention_mask"])


def test_truncation_and_drop_empty():
    msgs = [{"role": "user", "content": "Q" * 50}, {"role": "assistant", "content": "A"}]
    f = tokenize_conversation(msgs, CharTokenizer(), max_length=10)
    assert len(f["input_ids"]) == 10
    assert tokenize_all([msgs], CharTokenizer(), max_length=10) == []  # 답변이 잘려 나가면 버림


def test_split_train_valid():
    train, valid = split_train_valid(list(range(20)), 0.1, seed=1)
    assert len(valid) == 2 and len(train) == 18 and set(train) | set(valid) == set(range(20))
    train, valid = split_train_valid([1, 2, 3], 0.1)
    assert len(valid) == 1 and len(train) == 2


def test_pad_batch():
    batch = pad_batch(
        [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
         {"input_ids": [4], "attention_mask": [1], "labels": [4]}],
        pad_token_id=0,
    )
    assert batch["input_ids"][1] == [4, 0, 0]
    assert batch["attention_mask"][1] == [1, 0, 0]
    assert batch["labels"][1] == [4, IGNORE_INDEX, IGNORE_INDEX]
