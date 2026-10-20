"""LoRA 학습 실행 진입점.

    python -m training.train --config configs/train.yaml

Colab 에서는 notebooks/train_lora_colab.ipynb 가 이 명령을 실행한다.
체크포인트가 있으면 이어서 학습한다 (Colab 세션이 끊겼을 때).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from chatbot.config import PROJECT_ROOT, load_chat_config
from training.data_preprocessing import pad_batch, split_train_valid, tokenize_all
from training.dataset import load_conversations
from training.lora import apply_lora, build_quantization_config


def load_train_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_system_prompt(cfg: dict[str, Any]) -> str:
    """train.yaml 에 없으면 서비스용 model.yaml 의 시스템 프롬프트를 쓴다 (학습/서비스 일치)."""
    prompt = (cfg.get("data") or {}).get("system_prompt")
    return prompt.strip() if prompt else load_chat_config().system_prompt


def project_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_datasets(cfg: dict[str, Any], tokenizer):
    data = cfg["data"]
    system_prompt = resolve_system_prompt(cfg)
    conversations = load_conversations(project_path(data["train_file"]), system_prompt)
    if data.get("valid_file"):
        train, valid = conversations, load_conversations(project_path(data["valid_file"]), system_prompt)
    else:
        train, valid = split_train_valid(conversations, float(data.get("valid_ratio", 0.1)), cfg["training"]["seed"])
    max_length = int(data.get("max_length", 1024))
    train_feats = tokenize_all(train, tokenizer, max_length)
    valid_feats = tokenize_all(valid, tokenizer, max_length)
    print(f"[data] 학습 {len(train_feats)}개 / 검증 {len(valid_feats)}개 대화 (system prompt: {system_prompt[:30]!r}...)")
    return train_feats, valid_feats


def warmup_kwargs(training_args_cls, ratio: float) -> dict:
    """transformers 4.x 는 warmup_ratio, 5.x 는 warmup_steps 에 비율(0~1)을 넣는다."""
    names = {f.name for f in dataclasses.fields(training_args_cls)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in names else {"warmup_steps": ratio}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sol_chat LoRA 학습")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--max-steps", type=int, default=-1, help="동작 확인용으로 스텝 수 제한")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    cfg = load_train_config(project_path(args.config))
    run_name, base_model, tcfg = cfg["run_name"], cfg["base_model"], cfg["training"]
    torch.manual_seed(tcfg["seed"])

    # 1) 토크나이저 + 데이터
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_feats, valid_feats = build_datasets(cfg, tokenizer)
    if not train_feats:
        print("[error] 학습할 대화가 없어요.", file=sys.stderr)
        return 1

    # 2) 베이스 모델 + LoRA
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    quant = build_quantization_config(cfg.get("quantization"))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32),
        device_map="auto" if cuda else None,
        quantization_config=quant,
    )
    model.config.use_cache = False  # gradient checkpointing 과 같이 쓰기 위해
    model = apply_lora(model, cfg["lora"], quantized=quant is not None,
                       gradient_checkpointing=bool(tcfg.get("gradient_checkpointing")))

    # 3) 학습
    checkpoint_dir = project_path(cfg["output"]["checkpoint_dir"]) / run_name
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(tcfg["epochs"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=int(tcfg["batch_size"]),
        per_device_eval_batch_size=int(tcfg["batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        lr_scheduler_type="cosine",
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_strategy="epoch" if valid_feats else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=bf16,
        fp16=cuda and not bf16,
        report_to="none",
        seed=int(tcfg["seed"]),
        remove_unused_columns=False,
        **warmup_kwargs(TrainingArguments, float(tcfg.get("warmup_ratio", 0.0))),
    )

    def collate(features):
        batch = pad_batch(features, tokenizer.pad_token_id)
        return {k: torch.tensor(v) for k, v in batch.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_feats,
        eval_dataset=valid_feats or None,
        data_collator=collate,
    )
    last = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if last:
        print(f"[train] 체크포인트에서 이어서 학습: {last}")
    result = trainer.train(resume_from_checkpoint=last)

    # 4) 어댑터 저장 (수십 MB) + 학습 설정/결과 기록
    adapter_dir = project_path(cfg["output"]["adapter_dir"]) / run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    shutil.copy(project_path(args.config), adapter_dir / "train.yaml")
    metrics = dict(result.metrics)
    if valid_feats:
        metrics.update(trainer.evaluate())
    (adapter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] 어댑터 저장: {adapter_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
