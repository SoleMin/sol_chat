# Report: sol_chat 챗봇 전체 설계 (API 챗봇 → 웹 서비스 → Colab LoRA 학습)

- id: 20261020-chatbot-architecture
- date: 2026-10-20
- plan: docs/plans/20261020-chatbot-architecture/PLAN.md

## Summary

Stage 1~3 코드를 모두 구현했다. OpenRouter(Gemini)로 대화하는 챗봇 코어와 CLI, FastAPI 서버와 스트리밍 웹 채팅 화면,
외부 공개용 보호장치(접속 비밀번호, 요청 수 제한, 대화 분리), Colab용 LoRA 학습·평가 코드와 노트북,
학습한 모델을 웹에서 쓰기 위한 두 경로(Colab vLLM 서빙, PC 직접 로드)까지 들어 있다.
실제 API 키, Colab GPU, 휴대폰 외부 접속이 필요한 확인은 사용자 몫으로 남았다.

## Changes

- chatbot/: config, model(백엔드 인터페이스), backends/api_backend, backends/local_backend, preprocessing, generation, postprocessing, conversation, cli
- server/: main, routes, schemas, security
- web/: index.html, style.css, app.js
- training/: dataset, data_preprocessing, lora, train, evaluate
- notebooks/: train_lora_colab.ipynb, serve_vllm_colab.ipynb
- configs/: model.yaml, server.yaml, train.yaml
- data/examples/: sample.jsonl(33개 대화), eval_questions.txt, README.md
- tests/: 9개 파일, 60개 테스트
- requirements.txt, requirements-train.txt, .env.example, .gitignore, scripts/verify.sh(pytest 추가)
- README.md: sol_chat 사용법으로 교체 (기존 하네스 설명은 docs/harness.md 로 이동)
- docs/current-state.md 갱신

## Verification

- verdict: PASS_WITH_NOTES
- commands: `bash scripts/verify.sh` -> pytest 60 passed, verify: OK
- 브라우저 자동 확인 (Playwright, 가짜 백엔드): 1280px / 390px 화면에서 비밀번호 오류 표시, 로그인, 스트리밍 답변, 멀티턴, 새 대화, 대화 다시 열기, 휴대폰 서랍 메뉴, 페이지 JS 오류 없음
- 학습 파이프라인 확인 (CPU, 작은 랜덤 Qwen2 모델 + 글자 단위 토크나이저): train → 어댑터 저장 → 재실행 시 체크포인트에서 이어 학습 → evaluate 결과 파일 생성(베이스/LoRA 답변이 서로 다름) → local 백엔드로 /api/chat, /api/chat/stream 응답
- acceptance:
  - AC-001/002: 가짜 백엔드로 멀티턴 전달·시스템 프롬프트 적용을 테스트로 확인. 실제 키로는 사용자 확인 필요
  - AC-003: 브라우저 자동 확인 완료 (가짜 백엔드)
  - AC-004: curl 로 JSON 응답 확인 (가짜 백엔드)
  - AC-005: 키 없음/잘못된 키/크레딧 부족을 사용자용 메시지로 변환하는 것 테스트로 확인
  - AC-006/007/008: 작은 모델로 흐름 확인. 실제 Colab GPU·Qwen 학습은 사용자 확인 필요
  - AC-009: 통과
  - AC-010/011: 비밀번호 401, 요청 제한 429, 클라이언트별 대화 분리 테스트로 확인. 실제 터널·휴대폰 접속은 사용자 확인 필요

## Deviations from plan

- D-004: `trl` SFTTrainer 대신 transformers `Trainer` + 직접 만든 라벨 마스킹(`training/data_preprocessing.py`)을 사용. trl API가 버전마다 자주 바뀌어 Colab 최신 버전에서 깨질 위험이 크고, "답변 부분만 학습"하는 원리를 코드에서 직접 볼 수 있어 학습 목적에 더 맞음. 의존성도 하나 줄었다.
- NFR-004: 요청 제한을 세 가지로 나눔 (채팅 메시지 분당 30, 전체 API 분당 120, 비밀번호 실패 분당 10). 하나로 묶었더니 화면이 대화 목록을 새로 고치는 요청만으로도 제한에 걸렸음.
- 서버 설정을 `configs/server.yaml` 로 분리 (PLAN 구조표에는 model.yaml, train.yaml 만 있었음).
- transformers 5.x 대응: `warmup_ratio` 가 없어져 버전에 따라 `warmup_steps` 에 비율을 넣도록 처리.

## Deferred / Remaining risks

- 사용자 수동 확인: 실제 OpenRouter 키로 CLI/웹 대화 (T-017, T-030), Cloudflare 터널로 휴대폰 접속, Colab GPU 학습과 vLLM 서빙 (T-046)
- PLAN.md TBD-1~3 (서빙 방식, 베이스 모델, 학습 데이터)는 제안안으로 구현만 해둠. 결정 후 configs 값만 바꾸면 된다.
- `google/gemini-2.5-flash`, `qwen/qwen-2.5-7b-instruct` 모델 ID는 OpenRouter 목록에서 확인 필요 (바뀌었으면 model.yaml 만 수정).
- serve_vllm_colab.ipynb 는 이 환경에서 GPU·vLLM 이 없어 실행해 보지 못함.
- 대화는 서버 메모리에만 저장 (재시작 시 사라짐) — 의도된 범위.

## Context Index

- docs/plans/20261020-chatbot-architecture/PLAN.md — contains: 설계와 결정 — use_for: 왜 이렇게 만들었는지
- docs/plans/20261020-chatbot-architecture/TASKS.md — contains: 태스크별 결과와 남은 확인 — use_for: 이어서 작업
- README.md — contains: 설치·실행·외부 접속·Colab 학습 — use_for: 사용법
- tests/ — contains: 단위 테스트 60개 — use_for: 수정 후 회귀 확인
