"""transformers 로 모델을 직접 로드하는 백엔드 (베이스 모델 + 선택적 LoRA 어댑터).

torch / transformers / peft 가 필요하다 (requirements-train.txt).
GPU 가 없으면 CPU 로 동작하지만 느리다. 학습 후 평가(training/evaluate.py)도 이 클래스를 쓴다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Thread

from chatbot.config import PROJECT_ROOT, GenerationParams, ModelEntry
from chatbot.model import BackendError, ChatBackend, Message


STREAM_TIMEOUT_SECONDS = 120  # 다음 글자가 이 시간 안에 안 나오면 실패로 처리


def _generation_error(e: BaseException) -> BackendError:
    if "out of memory" in str(e).lower():
        return BackendError("로컬 모델 생성 중 메모리가 부족해요. 더 작은 모델이나 짧은 입력을 써주세요.")
    return BackendError(f"로컬 모델 생성 중 오류가 났어요: {type(e).__name__}")


def load_model(base_model: str, adapter_path: str = "", quantization_config=None):
    """(model, tokenizer) 를 돌려준다. adapter_path 가 있으면 LoRA 어댑터를 붙인다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype, device_map = torch.float32, None

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map=device_map, quantization_config=quantization_config
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


class LocalBackend(ChatBackend):
    def __init__(self, entry: ModelEntry, model=None, tokenizer=None):
        if model is None:
            if not entry.base_model:
                raise BackendError(f"모델 '{entry.name}'에 base_model 이 없어요.", status_code=500)
            try:
                adapter = entry.adapter_path
                if adapter and not Path(adapter).is_absolute():
                    adapter = str(PROJECT_ROOT / adapter)  # 상대 경로는 프로젝트 폴더 기준
                model, tokenizer = load_model(entry.base_model, adapter)
            except ImportError as e:
                raise BackendError(
                    "로컬 모델을 쓰려면 requirements-train.txt 를 설치해야 해요.", status_code=500
                ) from e
        self.entry = entry
        self.model = model
        self.tokenizer = tokenizer

    def _inputs(self, messages: list[Message]):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _generate_kwargs(self, params: GenerationParams) -> dict:
        kwargs = {"max_new_tokens": params.max_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if params.temperature > 0:
            kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def generate(self, messages, params):
        import torch

        try:
            inputs = self._inputs(messages)
            with torch.no_grad():
                output = self.model.generate(**inputs, **self._generate_kwargs(params))
        except Exception as e:
            raise _generation_error(e) from e
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stream(self, messages, params) -> Iterator[str]:
        from queue import Empty

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        try:
            kwargs = {**self._inputs(messages), **self._generate_kwargs(params), "streamer": streamer}
        except Exception as e:
            raise _generation_error(e) from e
        errors: list[BaseException] = []

        def run() -> None:
            # 생성 스레드에서 난 예외를 기록하고, 성공/실패 모두 스트림을 끝낸다
            # (안 그러면 받는 쪽이 영원히 기다린다)
            try:
                self.model.generate(**kwargs)
            except BaseException as e:  # noqa: BLE001 - 어떤 오류든 소비자에게 전달
                errors.append(e)
                streamer.end()

        worker = Thread(target=run, daemon=True)
        worker.start()
        try:
            yield from streamer
        except Empty as e:
            raise BackendError("로컬 모델 응답이 너무 오래 걸려요.", status_code=504) from e
        worker.join(timeout=1)
        if errors:
            raise _generation_error(errors[0])
