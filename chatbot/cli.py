"""터미널에서 챗봇과 대화하기.

    python -m chatbot.cli
    python -m chatbot.cli --model gemini-flash

명령: /reset 새 대화, /model <이름> 모델 바꾸기, /models 목록, exit 종료
"""

from __future__ import annotations

import argparse
import sys

from chatbot.config import ConfigError, load_chat_config
from chatbot.generation import ChatGenerator
from chatbot.model import BackendError, Message
from chatbot.postprocessing import clean_response
from chatbot.preprocessing import validate_user_input


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sol_chat 터미널 대화")
    parser.add_argument("--config", help="configs/model.yaml 경로")
    parser.add_argument("--model", help="사용할 모델 이름 (model.yaml 의 models 키)")
    args = parser.parse_args(argv)

    try:
        config = load_chat_config(args.config)
        model = config.get_model(args.model).name
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 1

    generator = ChatGenerator(config)
    history: list[Message] = []
    print(f"sol_chat ({config.models[model].label}) — /reset 새 대화, /models 목록, exit 종료\n")

    while True:
        try:
            line = input("나 > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command = line.strip()
        if command in ("exit", "quit", "/exit"):
            return 0
        if command == "/reset":
            history = []
            print("(새 대화를 시작합니다)\n")
            continue
        if command == "/models":
            for name, entry in config.models.items():
                print(f"  {'*' if name == model else ' '} {name}: {entry.label}")
            continue
        if command.startswith("/model "):
            try:
                model = config.get_model(command.split(maxsplit=1)[1]).name
                print(f"(모델 변경: {config.models[model].label})\n")
            except ConfigError as e:
                print(f"[오류] {e}\n")
            continue

        try:
            message = validate_user_input(line)
        except ValueError:
            continue

        print("솔챗 > ", end="", flush=True)
        parts: list[str] = []
        try:
            for chunk in generator.stream_reply(model, history, message):
                parts.append(chunk)
                print(chunk, end="", flush=True)
        except BackendError as e:
            print(f"\n[오류] {e.user_message}\n")
            continue
        print("\n")
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": clean_response("".join(parts))}]


if __name__ == "__main__":
    sys.exit(main())
