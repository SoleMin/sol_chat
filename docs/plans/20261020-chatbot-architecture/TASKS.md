# Tasks: sol_chat 챗봇 전체 설계 (API 챗봇 → 웹 서비스 → Colab LoRA 학습)

- plan: docs/plans/20261020-chatbot-architecture/PLAN.md
- status: IN_PROGRESS (코드 구현 완료, 사용자 수동 확인·Stage 3 결정 대기)

실행은 로드맵 Stage 단위로 끊는다. 각 Stage 끝의 검증 태스크가 통과하고 사용자가 동작을 확인하면 다음 Stage로 넘어간다.
(PLAN.md의 "Phase 1~3" 로드맵 = 아래 "Stage 1~3". 하네스 템플릿의 테스트→구현→검증 순서는 각 Stage 안에서 지킨다.)

---

## Stage 1: 기본 챗봇 코어 (OpenRouter API)

### 1-A. 기반 / 계약

- [x] T-001 [P] `requirements.txt` 작성 — fastapi, uvicorn, openai, pydantic, pyyaml, python-dotenv, pytest, httpx (NFR-001)
  - verify: Python 3.11 venv에서 `pip install -r requirements.txt` 성공
  > done: requirements.txt
- [x] T-002 [P] `.gitignore`에 `.env`, `.venv/`, `data/*`(단 `data/examples/` 제외), `outputs/`, `__pycache__/` 추가 (NFR-002)
  - verify: `git check-ignore .env outputs/x` 가 둘 다 출력
  > done: .gitignore
- [x] T-003 [P] `.env.example` 작성 — `OPENROUTER_API_KEY=`, `ACCESS_CODE=` (NFR-002)
  > done: .env.example
- [x] T-004 [P] `configs/model.yaml` 작성 — backend: api, base_url, model(Gemini 계열), system_prompt, temperature/top_p/max_tokens, max_history_turns (FR-003)
  > done: configs/model.yaml (+ configs/server.yaml 분리)
- [x] T-005 `chatbot/config.py` — model.yaml + .env 로드 → dataclass, 키 없으면 명확한 예외 (FR-003, AC-005) after: T-004
  - verify: `pytest tests/test_config.py`
  > done: chatbot/config.py, tests/test_config.py 통과
- [x] T-006 `chatbot/model.py` — 백엔드 인터페이스(`generate`, `stream`) + `create_backend(config)` 팩토리 + 테스트용 FakeBackend (FR-007, D-002) after: T-005
  > done: chatbot/model.py

### 1-B. 테스트 먼저

- [x] T-007 [P] `tests/test_preprocessing.py` — 시스템 프롬프트 포함, 히스토리 순서, 최대 턴 초과 시 오래된 턴부터 잘림, 공백/빈 입력 처리 (FR-002)
  > done: tests/test_preprocessing.py
- [x] T-008 [P] `tests/test_conversation.py` — 세션 생성/조회/목록, 클라이언트별 분리, 세션 수 상한·오래된 세션 정리 (FR-012, D-008)
  > done: tests/test_conversation.py
- [x] T-009 [P] `tests/test_postprocessing.py` — 앞뒤 공백·불필요한 토큰 정리 (FR-001)
  > done: tests/test_postprocessing.py

### 1-C. 구현

- [x] T-010 [P] `chatbot/preprocessing.py` — 입력 정리, 히스토리 → messages 변환, 길이 자르기 (FR-002) after: T-007
  - verify: `pytest tests/test_preprocessing.py`
  > done: chatbot/preprocessing.py, 테스트 통과
- [x] T-011 [P] `chatbot/conversation.py` — 메모리 세션 저장소 (FR-002, FR-012, D-008) after: T-008
  - verify: `pytest tests/test_conversation.py`
  > done: chatbot/conversation.py, 테스트 통과
- [x] T-012 [P] `chatbot/postprocessing.py` (FR-001) after: T-009
  - verify: `pytest tests/test_postprocessing.py`
  > done: chatbot/postprocessing.py, 테스트 통과
- [x] T-013 `chatbot/backends/api_backend.py` — openai SDK + OpenRouter base_url, 일반/스트리밍 호출, 인증·한도 오류를 공통 예외로 변환 (FR-001, AC-005) after: T-006
  > done: chatbot/backends/api_backend.py, tests/test_api_backend.py 통과 (가짜 클라이언트)
- [x] T-014 `chatbot/generation.py` — preprocessing → backend → postprocessing 연결, 일반/스트리밍 (FR-001, FR-003) after: T-010, T-012, T-013
  - verify: `pytest tests/test_generation.py` (FakeBackend 사용)
  > done: chatbot/generation.py, tests/test_generation.py 통과
- [x] T-015 `chatbot/cli.py` — 터미널 멀티턴 대화, `exit`로 종료, `/reset`로 새 대화 (FR-004) after: T-011, T-014
  > done: chatbot/cli.py, 가짜 백엔드로 대화·/reset·/models 확인, 키 없을 때 안내 메시지 확인

### 1-D. 검증

- [x] T-016 `scripts/verify.sh`에 `pytest -q` 추가 (기존 TS 체크 유지) (AC-009)
  - verify: `bash scripts/verify.sh` → `verify: OK`
  > done: scripts/verify.sh, verify: OK
- [ ] T-017 수동 QA: AC-001(멀티턴 기억), AC-002(시스템 프롬프트 변경 반영) — 사용자 PC에서 실제 키로 확인
  > 대기: 사용자 PC에서 실제 OPENROUTER_API_KEY 로 확인 필요

---

## Stage 2: 서비스화 (서버 + 웹 화면 + 외부 접속)

### 2-A. 테스트 먼저

- [x] T-018 [P] `tests/test_api.py` — FakeBackend로 `/api/chat` JSON 응답·session_id, `/api/chat/stream` SSE 형식, `/api/sessions` 클라이언트별 분리 (FR-005, FR-012, AC-004)
  > done: tests/test_api.py
- [x] T-019 [P] `tests/test_security.py` — 비밀번호 없으면 401, 틀리면 401, 맞으면 통과, 분당 제한 초과 시 429, 너무 긴 메시지 413/422 (FR-011, NFR-004, AC-010, AC-011)
  > done: tests/test_security.py

### 2-B. 구현

- [x] T-020 `server/schemas.py` — ChatRequest/ChatResponse/SessionSummary 등 (FR-005)
  > done: server/schemas.py
- [x] T-021 `server/security.py` — ACCESS_CODE 검사 의존성, 클라이언트/IP별 요청 수 제한, 메시지 길이 제한 (FR-011, NFR-004, D-009) after: T-019
  - verify: `pytest tests/test_security.py`
  > done: server/security.py (채팅/전체/비밀번호 실패 3종 제한), 테스트 통과
- [x] T-022 `server/routes.py` — `/api/chat`, `/api/chat/stream`(SSE), `/api/sessions`, `/api/sessions/{id}`, `/api/models`, `/api/auth/check` (FR-005, FR-007, FR-012) after: T-020, T-021
  - verify: `pytest tests/test_api.py`
  > done: server/routes.py, tests/test_api.py 통과
- [x] T-023 `server/main.py` — 앱 생성, 시작 시 백엔드 1회 초기화, `web/` 정적 서빙, 백엔드 오류 → 사용자용 오류 응답 (FR-005, AC-005) after: T-022
  > done: server/main.py
- [x] T-024 [P] `web/index.html` — 레이아웃: 대화 목록 사이드바, 채팅 영역, 입력창, 모델 선택, 접속 비밀번호 입력 화면 (FR-006, FR-011)
  > done: web/index.html
- [x] T-025 [P] `web/style.css` — 채팅 말풍선, 반응형(휴대폰 화면 대응) (FR-006, AC-010)
  > done: web/style.css (다크 모드, 휴대폰 서랍 메뉴)
- [x] T-026 `web/app.js` — 비밀번호 입력/저장, 클라이언트 ID 생성, 메시지 전송, SSE 스트리밍 표시, 새 대화/대화 전환, 오류 표시 (FR-006, FR-011, FR-012) after: T-022, T-024
  > done: web/app.js, 브라우저 자동 확인

### 2-C. 외부 접속 / 문서

- [x] T-027 `README.md`에 실행 방법 추가 — venv 생성(드라이브 밖), 설치, `.env` 설정, CLI, 서버 실행, Cloudflare 임시 터널(`cloudflared tunnel --url http://localhost:8000`), OpenRouter 키 사용 한도 설정 권장 (D-009)
  > done: README.md (하네스 설명은 docs/harness.md 로 이동)
- [x] T-028 `docs/current-state.md` 프로젝트 설명·실행 방법 갱신
  > done: docs/current-state.md

### 2-D. 검증

- [x] T-029 `bash scripts/verify.sh` 통과 (AC-009)
  > done: verify: OK (pytest 60 passed)
- [ ] T-030 수동 QA: AC-003(브라우저 스트리밍·새 대화·전환), AC-004(curl), AC-005(잘못된 키 오류 표시), AC-010(휴대폰 LTE로 터널 주소 접속, 비밀번호, 대화 분리), AC-011(요청 제한)
  > 부분 확인: 가짜 백엔드로 브라우저(PC 1280px, 휴대폰 390px) 로그인·스트리밍·새 대화·대화 전환 자동 확인 완료. 실제 키·터널·휴대폰 LTE 접속은 사용자 확인 필요

---

## Stage 3: Colab LoRA 학습

### 3-A. 착수 전 결정 (사용자와 함께, PLAN.md 반영 후 진행)

- [ ] T-031 TBD-2 결정: 학습할 베이스 모델 — OpenRouter로 후보 모델을 미리 써 보고 Colab GPU 메모리 기준으로 선택 → PLAN.md D-002/train.yaml 반영
  > 제안안으로 진행: Qwen/Qwen2.5-1.5B-Instruct (configs/train.yaml). 사용자 결정 대기
- [ ] T-032 TBD-3 결정: 학습 데이터 — 원하는 말투·기능, 데이터 수집 방법 → PLAN.md 반영
  > 제안안으로 진행: 예시 데이터 33개(반말 + '📌 한 줄 요약' 스타일). 사용자 결정 대기
- [ ] T-033 TBD-1 결정: 학습 모델 서빙 방식(D-006 a/b/c) → PLAN.md 반영
  - 위 세 결정 후 아래 태스크를 필요하면 수정 (범위가 바뀌면 사용자 승인)
  > 제안안으로 진행: (a) Colab vLLM 노트북 + (b) local_backend 둘 다 구현. 사용자 결정 대기

### 3-B. 데이터 / 학습 코드 (after: T-031, T-032)

- [x] T-034 [P] `data/examples/sample.jsonl` — messages 형식 예시 대화 수십 건 (D-007, AC-006)
  > done: data/examples/sample.jsonl (33개), eval_questions.txt, README.md
- [x] T-035 [P] `requirements-train.txt` — torch, transformers, peft, trl, datasets, accelerate, bitsandbytes (D-004)
  > done: requirements-train.txt
- [x] T-036 [P] `configs/train.yaml` — base_model, LoRA(r, alpha, dropout, target_modules), 학습 파라미터, 데이터/출력 경로 (FR-008)
  > done: configs/train.yaml
- [x] T-037 `tests/test_dataset.py` + `training/dataset.py` — JSONL 로드, 형식 검증(role 순서, 빈 내용) (FR-008)
  > done: training/dataset.py, tests/test_training_data.py 통과
- [x] T-038 `training/data_preprocessing.py` — chatbot/preprocessing의 messages 변환 재사용, chat template 적용, 답변 부분만 loss, train/valid 분할 (FR-008) after: T-037
  > done: training/data_preprocessing.py, 답변 마스킹 테스트 통과
- [x] T-039 `training/lora.py` — train.yaml → LoraConfig (+ 선택적 4bit 설정) (FR-008, D-004) after: T-036
  > done: training/lora.py
- [x] T-040 `training/train.py` — SFTTrainer 실행, Drive에 체크포인트·어댑터 저장 (FR-008) after: T-038, T-039
  > done: training/train.py, 작은 모델로 학습·저장·이어학습 확인
- [x] T-041 `training/evaluate.py` — 질문 목록으로 베이스 vs 어댑터 답변 비교 파일 저장 (FR-009) after: T-040
  > done: training/evaluate.py, 작은 모델로 결과 파일 생성 확인
- [x] T-042 `notebooks/train_lora_colab.ipynb` — Drive 마운트 → 설치 → train → evaluate (FR-008, D-005) after: T-040, T-041
  > done: notebooks/train_lora_colab.ipynb

### 3-C. 서비스 연결 (after: T-033)

- [x] T-043 `chatbot/backends/local_backend.py` 및/또는 Colab vLLM 서빙 셀 — T-033 결정에 따라 구현 (FR-007, FR-010)
  > done: chatbot/backends/local_backend.py + notebooks/serve_vllm_colab.ipynb, 로컬 백엔드로 /api/chat·stream 확인
- [x] T-044 `configs/model.yaml`에 학습 모델 항목 추가, `/api/models`·웹 모델 선택에 노출 (FR-007)
  > done: configs/model.yaml 에 qwen-base / my-lora-colab / my-lora-local 예시(주석)

### 3-D. 검증

- [x] T-045 `bash scripts/verify.sh` 통과 (AC-009)
  > done: verify: OK
- [ ] T-046 수동 QA: AC-006(Colab 학습·어댑터 생성), AC-007(비교 결과), AC-008(웹에서 학습 모델 대화)
  > 부분 확인: 작은 랜덤 Qwen2 모델(CPU)로 학습→어댑터 저장→체크포인트 이어학습→evaluate→웹 API 대화까지 동작 확인. 실제 Colab GPU 학습은 사용자 확인 필요
- [x] T-047 Wrapup: `docs/reports/20261020-chatbot-architecture-REPORT.md` + `docs/current-state.md` 갱신
  > done: docs/reports/20261020-chatbot-architecture-REPORT.md, docs/current-state.md

<!--
Conventions:
- [P] = parallelizable (disjoint file scope from other [P] tasks in the phase)
- after: T-xxx = dependency
- Execution appends one result line under each finished task:
  > done: <files>, <verify result>
-->
