# Plan: sol_chat 챗봇 전체 설계 (API 챗봇 → 웹 서비스 → Colab LoRA 학습)

- id: 20261020-chatbot-architecture
- status: APPROVED (rev 4, 2026-10-20 사용자 승인)
- date: 2026-10-20

## 변경 이력

- rev 4 (2026-10-20): 외부 접속 가능하도록 범위 확장(FR-011, D-009). 대화 기록은 서버 메모리 저장 확정. 화면 기술 확정. 학습 모델 서빙 방식·베이스 모델·학습 데이터는 Phase 3 착수 전 결정 태스크로 분리.
- rev 3 (2026-10-20): API 제공자를 OpenRouter로 확정(모델은 Gemini 계열). Python 3.11 + venv, 한국어 위주로 확정. Open Questions 해소.
- rev 2 (2026-10-20): 시작 단계를 "사전학습 모델 직접 로드"에서 "외부 LLM API 호출"로 변경. 학습은 Google Colab GPU에서 진행하는 것으로 확정. 로컬 GPU 질문 삭제.
- rev 1 (2026-10-20): 초안.

## Intent

AI 모델링 실력을 키우기 위한 토이 프로젝트로, 먼저 **외부 LLM API를 불러 쓰는 가장 기본적인 챗봇**을 만들고, 이를 프롬프트/터미널이 아닌 **실제 서비스처럼 쓸 수 있는 웹 화면**에서 동작시킨다. 이후 **Colab GPU에서 오픈 모델을 LoRA로 학습**시켜 원하는 성격·기능의 챗봇으로 바꿔 나간다. 코드는 나중에 다시 봐도 이해하기 쉽도록 역할별 파일(model.py, preprocessing.py 등)과 폴더(서비스 / 챗봇 코어 / 학습 / 화면)로 분명하게 나눈다.

숨은 의도로 읽은 것:
- **"API로 시작"과 "나중에 내가 학습한 모델"이 같은 화면·같은 코드 흐름에서 갈아끼워져야 한다.** 그래서 `chatbot/model.py`는 "모델 백엔드"를 인터페이스로 두고, 1단계는 API 백엔드, 3단계는 학습된 모델 백엔드를 구현한다. 설정 한 줄로 전환하고 같은 질문으로 비교할 수 있어야 한다.
- 가능한 한 **OpenAI 호환 API 형식(chat completions)**에 맞춘다. 많은 API 제공자(OpenAI, Gemini, Hugging Face Inference 등)가 이 형식을 지원하고, 3단계에서 Colab에 띄운 LoRA 모델도 vLLM으로 같은 형식의 서버로 띄울 수 있다. 그러면 학습한 모델로 바꿀 때 서비스 코드는 거의 그대로다.
- 학습은 API 모델이 아니라 **공개 가중치가 있는 오픈 모델**(예: Qwen 계열 소형 instruct)을 대상으로 한다. 외부 API 모델(GPT, Claude, Gemini 등)은 가중치가 없어 직접 LoRA를 걸 수 없기 때문이다.

## Problem / Why

현재 레포에는 개발 하네스(규칙, 서브에이전트, 훅)만 있고 챗봇 코드는 없다. 이후 단계를 반복 확장하려면 처음에 폴더 구조와 모듈 경계를 정해 두는 것이 가장 비용이 적다.

## 단계별 로드맵

| Phase | 목표 | 결과물 |
| --- | --- | --- |
| 1. 기본 챗봇 코어 | 외부 LLM API로 멀티턴 대화 | `chatbot/` 패키지, CLI로 대화 확인 |
| 2. 서비스화 | API 서버 + 웹 채팅 화면 | `server/`, `web/`, 브라우저에서 스트리밍 대화 |
| 3. Colab LoRA 학습 | 오픈 모델을 내 데이터로 LoRA 학습, 학습 모델로 서비스 | `training/`, `notebooks/`, 웹에서 "API 모델 / 내 모델" 전환 |
| 4. 확장 (나중) | 원하는 기능 추가 | 별도 PLAN으로 진행 |

이번 PLAN의 승인 범위는 **전체 구조 + Phase 1~3**이다. 실행(/tasks, /execute)은 Phase 단위로 끊어서 진행하고, 각 Phase 끝에 동작 확인 후 다음으로 넘어간다. Phase 3의 세부는 아래 "Phase 3 착수 전 결정 사항"에 모아 두고, Phase 3 태스크의 첫 단계에서 사용자와 함께 정한다.

## 폴더 / 파일 구조

```
sol_chat/
├─ chatbot/                  # 챗봇 코어 (서비스·학습 양쪽에서 공용)
│  ├─ config.py              # 설정 로드 (configs/*.yaml + .env → dataclass)
│  ├─ model.py               # 모델 백엔드 인터페이스 + 백엔드 선택(팩토리)
│  ├─ backends/
│  │  ├─ api_backend.py      # Phase 1: OpenRouter API 호출 (OpenAI 호환 형식)
│  │  └─ local_backend.py    # Phase 3: transformers + LoRA 어댑터 로드 (Colab/GPU용)
│  ├─ preprocessing.py       # 입력 정리, 시스템 프롬프트 + 대화 히스토리 → messages 변환, 길이 자르기
│  ├─ generation.py          # 생성 호출 (일반 / 스트리밍), 생성 파라미터 적용
│  ├─ postprocessing.py      # 출력 정리 (공백·특수 토큰 정리 등)
│  ├─ conversation.py        # 대화 세션/히스토리 관리 (메모리 저장)
│  └─ cli.py                 # 터미널 대화용 (Phase 1 확인용)
├─ server/                   # 서비스 백엔드
│  ├─ main.py                # FastAPI 앱 생성, 백엔드 1회 초기화, 정적 웹 파일 서빙
│  ├─ routes.py              # /api/chat, /api/chat/stream, /api/sessions, /api/models
│  └─ schemas.py             # 요청/응답 데이터 형식 (pydantic)
├─ web/                      # 프론트엔드 (빌드 도구 없는 순수 HTML/CSS/JS)
│  ├─ index.html             # 채팅 화면
│  ├─ style.css
│  └─ app.js                 # 메시지 전송, 스트리밍 수신, 대화 목록, 새 대화, 모델 선택
├─ training/                 # 학습 코드 (Phase 3, Colab에서 실행)
│  ├─ dataset.py             # 원본 데이터 로드 → messages 형식 대화로 변환
│  ├─ data_preprocessing.py  # chat template 적용, 토크나이즈, 답변 부분만 loss, train/valid 분할
│  ├─ lora.py                # LoRA 설정 (r, alpha, dropout, target_modules) 생성
│  ├─ train.py               # 학습 실행 진입점 (python -m training.train)
│  └─ evaluate.py            # 같은 질문에 대해 베이스 vs 어댑터 답변 비교 저장
├─ notebooks/
│  └─ train_lora_colab.ipynb # Colab용: Drive 마운트 → 패키지 설치 → training.train 실행 → 어댑터 저장
├─ configs/
│  ├─ model.yaml             # 사용할 백엔드, 모델 이름, 생성 파라미터, 시스템 프롬프트
│  └─ train.yaml             # 베이스 모델, LoRA/학습 하이퍼파라미터, 데이터 경로
├─ data/                     # 학습 데이터 (raw/, processed/) — git 제외, 예시 파일만 커밋
├─ outputs/                  # 학습된 어댑터, 로그 — git 제외
├─ tests/                    # pytest (실제 API/모델 없이 도는 단위 테스트)
├─ .env.example              # API 키 이름 예시 (실제 키는 .env, git 제외)
├─ requirements.txt          # 서비스용 (가벼움: fastapi, openai SDK 등)
├─ requirements-train.txt    # 학습용 (torch, transformers, peft, trl — Colab에서만 설치)
└─ (기존 하네스 파일: .claude/, scripts/, docs/, AGENTS.md ...)
```

원칙:
- `chatbot/`은 웹이나 학습을 모른다. `server/`와 `training/`이 `chatbot/`을 가져다 쓴다.
- 대화 → messages 변환은 `chatbot/preprocessing.py` 한 곳에서만 한다. 학습 데이터 형식과 서비스 입력 형식이 어긋나지 않게 하기 위함이다.
- 서비스용 패키지와 학습용 패키지를 분리해 PC에는 무거운 torch를 설치하지 않아도 된다.

## Requirements

FR-001: `configs/model.yaml`에 적힌 모델(기본: OpenRouter의 Gemini 계열)을 `.env`의 `OPENROUTER_API_KEY`로 호출해 한국어 질문에 답한다.
FR-002: 멀티턴 대화를 지원한다. 이전 대화가 입력에 포함되고, 설정한 최대 턴/길이를 넘으면 오래된 턴부터 잘린다.
FR-003: 시스템 프롬프트, 모델 이름, 생성 파라미터(temperature, top_p, max_tokens)를 코드 수정 없이 설정 파일로 바꿀 수 있다.
FR-004: `python -m chatbot.cli`로 터미널에서 대화할 수 있다 (Phase 1 확인용).
FR-005: FastAPI 서버가 채팅 API를 제공한다. 일반 응답과 스트리밍(SSE) 응답 둘 다.
FR-006: 웹 화면에서 메시지 입력, 답변이 흘러나오는 스트리밍 표시, 새 대화 시작, 대화 목록 전환이 된다.
FR-007: 모델 백엔드를 설정으로 전환할 수 있다 (`api` / `local`). 웹 화면에서 설정에 등록된 모델 중 하나를 골라 대화할 수 있다.
FR-008: Colab 노트북에서 `training.train`을 실행해 messages 형식 JSONL 데이터로 오픈 모델을 LoRA 학습하고, 어댑터를 Google Drive의 `outputs/adapters/<run_name>/`에 저장한다.
FR-009: `training/evaluate.py`가 같은 질문 목록에 대해 베이스 모델과 어댑터 모델의 답변을 나란히 파일로 저장한다.
FR-010: 학습된 모델을 서비스에서 쓰는 경로를 하나 이상 제공한다 (아래 D-006).
NFR-001: 파일 하나는 한 가지 역할만 맡는다 (위 구조표의 설명 한 줄로 요약 가능해야 함).
NFR-002: API 키는 `.env`에만 두고 git에 올라가지 않는다. 학습 데이터, 어댑터, 가상환경도 git 제외.
FR-011: 외부(다른 기기, 다른 네트워크)에서 웹 주소로 접속해 채팅할 수 있다. 접속하려면 접속 비밀번호(access code)가 필요하다.
FR-012: 접속자마다 대화가 분리된다. 다른 사람의 대화 목록·내용은 보이지 않는다 (브라우저별 클라이언트 ID 기준).
NFR-004: 외부 공개 시 API 크레딧 남용을 막는다. 접속 비밀번호, IP/클라이언트별 분당 요청 수 제한, 메시지 길이 제한을 둔다.
NFR-003: 단위 테스트는 실제 API 호출이나 모델 다운로드 없이 돈다 (가짜 백엔드 사용).

## Acceptance Criteria

AC-001: `.env`에 `OPENROUTER_API_KEY`를 넣고 `python -m chatbot.cli` 실행 → "안녕, 너는 누구야?"에 한국어 답변, 이어서 "방금 내가 뭐라고 했지?"에 이전 질문을 반영한 답변. (FR-001, FR-002, FR-004)
AC-002: `configs/model.yaml`의 시스템 프롬프트를 바꾸고 재시작하면 답변 말투가 바뀐다. (FR-003)
AC-003: `uvicorn server.main:app` 실행 후 브라우저에서 `http://localhost:8000` 접속 → 메시지를 보내면 답변이 스트리밍으로 표시되고, "새 대화" 후 이전 대화로 돌아갈 수 있다. (FR-005, FR-006, 수동 QA)
AC-004: `curl -X POST localhost:8000/api/chat -H "Content-Type: application/json" -d '{"session_id":null,"message":"안녕"}'`가 JSON으로 답변과 session_id를 돌려준다. (FR-005)
AC-005: API 키가 없거나 틀리면 서버가 죽지 않고 화면에 이해할 수 있는 오류 메시지가 뜬다. (FR-001, 수동 QA)
AC-006: Colab에서 `notebooks/train_lora_colab.ipynb`를 위에서부터 실행하면 예시 데이터(`data/examples/sample.jsonl`)로 학습이 끝까지 돌고 Drive의 `outputs/adapters/...`에 `adapter_config.json`, `adapter_model.safetensors`가 생긴다. (FR-008)
AC-007: `evaluate.py` 결과 파일에서 같은 질문에 대한 베이스/어댑터 답변을 비교할 수 있고, 학습 데이터의 말투·형식이 어댑터 쪽에 반영된다. (FR-009)
AC-008: 학습된 모델을 D-006 방식으로 띄우고 웹 화면에서 그 모델을 선택해 대화할 수 있다. (FR-007, FR-010)
AC-010: 휴대폰(LTE 등 다른 네트워크)에서 공개 주소로 접속 → 비밀번호 없이는 채팅 API가 401로 거부되고, 비밀번호 입력 후에는 채팅이 된다. 다른 브라우저에서 접속하면 서로의 대화 목록이 보이지 않는다. (FR-011, FR-012, 수동 QA)
AC-011: 분당 요청 제한을 넘기면 429와 안내 메시지가 뜬다. (NFR-004)
AC-009: `bash scripts/verify.sh`가 기존 TS 타입체크 + `pytest`를 실행하고 통과한다. (NFR-003)

## Technical Decisions

D-001: 1단계 모델은 외부 LLM API, 호출 형식은 OpenAI 호환 chat completions
- Rationale: 로컬 GPU 없이 바로 품질 좋은 대화가 됨. OpenAI 호환 형식은 여러 제공자와 vLLM이 공통으로 지원해서 3단계에서 학습 모델로 바꿀 때 `api_backend.py`를 그대로 재사용할 수 있음 (base_url만 변경).
- Linked: FR-001, FR-007, FR-010
- 제공자: **OpenRouter** (`base_url: https://openrouter.ai/api/v1`, 키는 `OPENROUTER_API_KEY`). 모델은 `model.yaml`의 `model` 값으로 지정하고 기본은 Gemini 계열(예: `google/gemini-2.5-flash`, 정확한 ID는 구현 시 OpenRouter 모델 목록에서 확인). 호출은 `openai` Python SDK에 base_url만 바꿔서 사용.
- OpenRouter를 쓰면 키 하나로 Gemini 외 다른 모델(GPT, Claude, Llama, Qwen 등)도 `model` 값만 바꿔 비교해 볼 수 있음. 나중에 학습할 오픈 베이스 모델(Qwen 등)의 원래 성능을 학습 전에 미리 확인하는 데도 쓸 수 있음.
- Rejected alternative: 처음부터 로컬에 사전학습 모델 로드 — 시작 부담이 큼. Gemini API 직접 호출 — 가능하지만 OpenRouter가 여러 모델을 같은 형식으로 묶어 줘서 실험에 더 유리함.

D-002: 모델 백엔드 인터페이스 (`generate(messages, params)`, `stream(messages, params)`)
- Rationale: API ↔ 학습 모델 전환을 설정 한 줄로 하고, 테스트에서는 가짜 백엔드를 끼울 수 있음.
- Linked: FR-007, NFR-003

D-003: 백엔드 FastAPI + 프론트 순수 HTML/CSS/JS, 스트리밍은 SSE
- Rationale: "실제 서비스 같은 화면" 요구에 맞게 API와 화면을 분리. 빌드 도구 없이 파일 3개라 구조가 한눈에 보임. 나중에 React 등으로 바꿔도 API는 그대로.
- Linked: FR-005, FR-006
- Rejected alternative: Gradio/Streamlit — 빠르지만 데모 느낌이고 API/화면 분리 구조를 배우기 어려움. React+Vite — 이 단계엔 과함.

D-004: 학습은 Colab GPU에서 `peft` LoRA + `trl` SFTTrainer, 메모리가 부족하면 4bit QLoRA(`bitsandbytes`)
- Rationale: 소형 모델 파인튜닝의 표준 방식이고 어댑터만 저장(수십 MB)해 실험을 여러 개 바꿔 끼우기 쉬움. Colab은 리눅스라 bitsandbytes도 문제없음.
- Linked: FR-008
- Rejected alternative: 전체 파인튜닝 — Colab GPU 메모리로 어렵고 실험 비용이 큼.

D-005: Colab ↔ 코드/데이터 연결은 Google Drive 마운트
- Rationale: 레포가 이미 `G:\내 드라이브\sol_chat`에 있어서, Colab에서 Drive를 마운트하면 같은 코드·데이터를 그대로 쓰고 어댑터도 Drive에 바로 저장됨. 프라이빗 레포 clone용 GitHub 토큰이 필요 없음.
- Linked: FR-008
- Rejected alternative: Colab에서 git clone — 프라이빗 레포라 토큰 관리가 필요하고, 결과물을 따로 옮겨야 함.

D-006: 학습한 모델을 서비스에 연결하는 방법 — **미정, Phase 3 착수 전 결정 (TBD-1)**
- (a) Colab에서 vLLM으로 OpenAI 호환 서버를 띄우고 터널(cloudflared/ngrok)로 주소를 열어, 로컬 서버의 `api_backend`가 그 주소를 호출 — 코드 변경이 가장 적음. Colab 세션이 살아 있는 동안만 동작.
- (b) 어댑터를 PC로 받아 `local_backend`로 CPU 추론 — 상시 가능하지만 느림 (소형 모델이면 체험용으로는 가능).
- (c) Hugging Face Hub에 어댑터를 올리고 추론 서비스 사용 — 편하지만 비용/제약 확인 필요.
- 현재 제안: (a)로 시작, (b)를 보조로 구현.
- Linked: FR-010

D-007: 학습 데이터 형식은 messages 형식 JSONL (`{"messages":[{"role":"user",...},{"role":"assistant",...}]}`)
- Rationale: API 입력 형식, 서비스 대화 히스토리, 오픈 모델 chat template이 모두 같은 모양이라 실제 대화 로그를 학습 데이터로 재활용하기 쉬움.
- Linked: FR-008

D-008: 대화 세션은 처음엔 서버 메모리에 저장
- Rationale: 기본형에 충분 (사용자 확정). 서버 재시작 시 사라지는 한계는 Phase 4에서 SQLite 저장으로 확장. 외부 접속자가 생기므로 세션 수·세션당 턴 수 상한과 오래된 세션 자동 정리를 둬서 메모리가 계속 늘지 않게 한다.
- Linked: FR-002, FR-006, FR-012

D-009: 외부 접속은 PC에서 서버를 띄우고 Cloudflare Tunnel로 공개 주소 연결 + 접속 비밀번호
- Rationale: 클라우드 서버 없이 무료로 HTTPS 공개 주소를 얻을 수 있고, 공유기 포트포워딩이 필요 없음. 비밀번호(`.env`의 `ACCESS_CODE`)는 서버 미들웨어에서 검사해, 주소가 퍼져도 크레딧이 남용되지 않게 함. 회원가입/로그인 같은 계정 시스템은 아직 두지 않음.
- 운영: 우선 빠른 임시 터널(`cloudflared tunnel --url http://localhost:8000`, 실행할 때마다 주소가 바뀜)로 시작. 고정 주소가 필요하면 Cloudflare 계정 + 도메인으로 named tunnel 전환 (README에 절차 정리).
- Linked: FR-011, NFR-004
- Rejected alternative: 클라우드 배포(Render, Fly.io, AWS 등) — 상시 접속은 되지만 설정·비용이 들고 이 단계엔 과함 (Phase 4 후보). ngrok — 비슷하지만 무료 등급 제약이 더 많음. 포트포워딩 — 공유기 설정과 보안 부담.

## Scope

### In scope
- 위 폴더 구조의 `chatbot/`, `server/`, `web/`, `training/`, `notebooks/`, `configs/`, `tests/`, requirements 파일, `.env.example` 신규 작성
- `.gitignore`에 `.env`, `data/` (예시 제외), `outputs/`, `.venv/` 추가
- `scripts/verify.sh`에 `pytest` 추가, `docs/current-state.md`의 프로젝트 설명 채우기
- README에 실행 방법(설치 → API 키 설정 → CLI → 서버 → 외부 공개(터널) → Colab 학습 → 학습 모델 연결) 추가
- `server/` 쪽에 접속 비밀번호 검사, 요청 수 제한, 세션 정리 (필요하면 `server/security.py`로 분리)

### Non-goals
- 회원가입/로그인(계정 시스템), 클라우드 서버 배포, 대화 DB 영구 저장
- RAG(문서 검색), 도구 호출, 음성 등 고급 기능 (Phase 4 이후 별도 PLAN)
- API 제공자의 자체 파인튜닝 기능 사용 (LoRA를 직접 다루는 것이 목적)
- 대규모 데이터 수집/정제, 모델 성능 벤치마크
- 기존 하네스(TypeScript 훅 등) 구조 변경

## Risks

- **API 비용/한도**: OpenRouter는 크레딧 충전 방식이고 모델마다 단가가 다름. `:free` 모델은 호출 한도가 낮고 바뀔 수 있음. → `max_tokens`와 히스토리 길이 제한을 설정에 두고, 테스트는 가짜 백엔드로.
- **외부 공개로 인한 크레딧 남용·공격**: 주소가 퍼지면 누구나 호출할 수 있음. → 접속 비밀번호, 요청 수/메시지 길이 제한, OpenRouter 대시보드에서 키별 사용 한도(credit limit) 설정 권장. 시스템 프롬프트 우회(프롬프트 인젝션)는 막기 어려우니 시스템 프롬프트에 비밀 정보를 넣지 않음.
- **PC가 꺼지면 서비스도 꺼짐**: 터널 방식이라 PC와 서버가 켜져 있어야 접속 가능. → 상시 운영이 필요해지면 클라우드 배포를 별도 PLAN으로.
- **API 키 유출**: 키가 커밋되거나 구글 드라이브로 공유될 수 있음. → `.env`는 gitignore, 키를 코드/설정 파일에 직접 쓰지 않음. 드라이브 폴더 공유 시 `.env` 주의.
- **레포가 구글 드라이브 폴더 안에 있음**: 가상환경(.venv)이 드라이브로 동기화되면 느리고 용량을 잡아먹음. → 가상환경은 드라이브 밖 경로에 생성. (반대로 Colab 연동에는 유리 — D-005)
- **Colab 세션 제한**: 무료 Colab은 GPU 사용 시간이 제한되고 세션이 끊기면 학습이 중단됨. → 체크포인트를 Drive에 주기적으로 저장, 작은 모델·적은 데이터로 먼저 검증.
- **API 모델과 학습 모델의 품질 차이**: 소형 오픈 모델 + LoRA는 상용 API 모델보다 일반 능력이 낮음. → 비교 기준을 "일반 지능"이 아니라 "내가 학습시킨 말투·형식이 반영됐는가"로 둠 (AC-007).
- **chat template 불일치**: 학습 때와 서비스 때 프롬프트 형식이 다르면 학습 효과가 안 보임. → 둘 다 `chatbot/preprocessing.py`의 messages 변환 사용.

## Phase 3 착수 전 결정 사항 (사용자 결정으로 보류)

Phase 1~2 진행에는 영향이 없다. Phase 3 TASKS.md의 첫 태스크로 만들어 사용자와 함께 정한다.

- TBD-1: 학습한 모델을 서비스에 연결하는 방식 (D-006의 a/b/c 중 선택)
- TBD-2: 학습할 베이스 모델 (현재 제안: Qwen 계열 1.5B~3B Instruct, Colab GPU 메모리와 OpenRouter로 미리 써 본 결과로 결정)
- TBD-3: 학습 데이터 — 어떤 말투·기능으로 학습시킬지, 데이터를 어떻게 모을지 (그 전까지는 동작 확인용 예시 데이터 수십 건만 사용)

## Open Questions

(없음)

- 해소: API 제공자 → OpenRouter, 모델은 Gemini 계열 (D-001)
- 해소: 대화 언어 → 한국어 위주
- 해소: Python 3.11 + venv, 가상환경은 드라이브 밖 경로
- 해소: 서비스 범위 → 외부 접속 가능 (D-009), 대화 기록 → 서버 메모리 (D-008), 화면 → FastAPI + 순수 HTML/JS (D-003)
