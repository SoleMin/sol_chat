"""LoRA / 양자화 설정 만들기 (configs/train.yaml -> peft / transformers 객체)."""

from __future__ import annotations

from typing import Any


def build_lora_config(lora_cfg: dict[str, Any]):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules") or ["q_proj", "v_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_quantization_config(quant_cfg: dict[str, Any] | None):
    """load_in_4bit: true 면 QLoRA 용 4bit 설정, 아니면 None."""
    if not (quant_cfg or {}).get("load_in_4bit"):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )


def apply_lora(model, lora_cfg: dict[str, Any], quantized: bool, gradient_checkpointing: bool):
    """베이스 모델에 LoRA 어댑터를 붙이고, 학습되는 파라미터 수를 출력한다."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, build_lora_config(lora_cfg))
    model.print_trainable_parameters()
    return model
