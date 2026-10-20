# Current State

> A new session starts here. Keep this short and current — it is the handoff
> packet between sessions.

## Project

sol_chat: OpenRouter(Gemini) API로 시작해, 웹 서비스 화면을 붙이고, Colab에서 LoRA로 직접 학습한 모델로 확장하는 AI 모델링 연습용 챗봇.

## Last completed work

- 20261020-chatbot-architecture: Stage 1~3 코드 전체 구현 (챗봇 코어, FastAPI + 웹 화면 + 외부 접속 보호장치, LoRA 학습/평가/서빙).
  보고서: docs/reports/20261020-chatbot-architecture-REPORT.md
- 20261020-critic-fixes: 외부 평가(docs/reports/20261020-baseline-CRITIC.md) B-001~B-006 수정 (동시성, 세션 보관 상한, 스트림 종료 판정, 인증 잠금).
  보고서: docs/reports/20261020-critic-fixes-REPORT.md

## In flight

- 사용자 수동 확인 대기: 실제 OpenRouter 키로 CLI/웹 대화, Cloudflare 터널로 휴대폰 접속, Colab 학습 실행 (TASKS.md T-017, T-030, T-046)
- Stage 3 결정 대기 (PLAN.md TBD-1~3): 학습 모델 서빙 방식, 베이스 모델, 학습 데이터. 현재는 제안안(vLLM 서빙 + Qwen2.5-1.5B-Instruct + 예시 데이터)으로 구현됨.

## How to verify

- `bash scripts/verify.sh` (pytest 69개, API 키·GPU 불필요)
- 실행: README.md 의 1~4절

## Context Index

- docs/plans/20261020-chatbot-architecture/ — contains: PLAN.md (설계, 결정 D-001~D-009, TBD), TASKS.md — use_for: 설계 의도와 남은 작업 확인
- docs/reports/ — contains: completed work reports — use_for: history and evidence
- README.md — contains: 설치·실행·외부 접속·Colab 학습 방법 — use_for: 사용법
- configs/ — contains: model.yaml / server.yaml / train.yaml — use_for: 동작 조정
