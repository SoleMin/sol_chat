"""학습 결과 비교: 같은 질문에 대해 베이스 모델 vs LoRA 어댑터 답변을 나란히 저장.

    python -m training.evaluate --config configs/train.yaml
    python -m training.evaluate --adapter outputs/adapters/sol-lora-v1 --questions data/examples/eval_questions.txt

결과: outputs/eval/<run_name>-<시각>.md (읽기용) + .jsonl (기록용)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from chatbot.backends.local_backend import LocalBackend, load_model
from chatbot.config import GenerationParams, ModelEntry
from chatbot.postprocessing import clean_response
from training.train import load_train_config, project_path, resolve_system_prompt


def read_questions(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [q.strip() for q in lines if q.strip() and not q.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="베이스 vs LoRA 답변 비교")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--adapter", help="기본: outputs/adapters/<run_name>")
    parser.add_argument("--questions", default="data/examples/eval_questions.txt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 이면 매번 같은 답 (비교용)")
    args = parser.parse_args(argv)

    cfg = load_train_config(project_path(args.config))
    adapter = project_path(args.adapter) if args.adapter else project_path(cfg["output"]["adapter_dir"]) / cfg["run_name"]
    if not adapter.exists():
        print(f"[error] 어댑터가 없어요: {adapter} (먼저 training.train 실행)", file=sys.stderr)
        return 1

    system_prompt = resolve_system_prompt(cfg)
    questions = read_questions(project_path(args.questions))
    params = GenerationParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_new_tokens)

    model, tokenizer = load_model(cfg["base_model"], str(adapter))
    entry = ModelEntry(name="eval", label="eval", backend="local", base_model=cfg["base_model"], adapter_path=str(adapter))
    backend = LocalBackend(entry, model=model, tokenizer=tokenizer)

    rows = []
    for i, q in enumerate(questions, 1):
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        with model.disable_adapter():  # 같은 모델에서 어댑터만 끄면 베이스 모델
            base = clean_response(backend.generate(messages, params))
        tuned = clean_response(backend.generate(messages, params))
        rows.append({"question": q, "base": base, "lora": tuned})
        print(f"[{i}/{len(questions)}] {q}")

    out_dir = project_path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{cfg['run_name']}-{datetime.now():%Y%m%d-%H%M%S}"
    with open(f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = [f"# 평가: {cfg['run_name']}", "", f"- base: `{cfg['base_model']}`", f"- adapter: `{adapter}`", ""]
    for row in rows:
        md += [f"## Q. {row['question']}", "", "**베이스**", "", row["base"], "", "**LoRA**", "", row["lora"], ""]
    Path(f"{stem}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] {stem}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
