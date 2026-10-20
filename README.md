# sol_chat (솔챗)

AI 모델링을 연습하기 위한 토이 챗봇 프로젝트입니다.

1. **Stage 1**: OpenRouter API(Gemini)로 동작하는 기본 챗봇
2. **Stage 2**: 실제 서비스처럼 쓰는 웹 채팅 화면 + 외부 접속
3. **Stage 3**: Colab GPU에서 오픈 모델을 LoRA로 학습해서, 같은 화면에서 내 모델과 대화

설계 문서: [`docs/plans/20261020-chatbot-architecture/PLAN.md`](docs/plans/20261020-chatbot-architecture/PLAN.md)

## 폴더 구조

| 폴더/파일 | 역할 |
| --- | --- |
| `chatbot/config.py` | 설정 로드 (`configs/*.yaml` + `.env`) |
| `chatbot/model.py` | 모델 백엔드 인터페이스, 설정에 맞는 백엔드 선택 |
| `chatbot/backends/api_backend.py` | OpenAI 호환 API 호출 (OpenRouter, Colab vLLM) |
| `chatbot/backends/local_backend.py` | transformers로 베이스 모델 + LoRA 어댑터 직접 로드 |
| `chatbot/preprocessing.py` | 입력 정리, 대화 → messages 변환, 히스토리 자르기 (학습에도 공용) |
| `chatbot/generation.py` | 전처리 → 모델 → 후처리 흐름 |
| `chatbot/postprocessing.py` | 출력 정리 |
| `chatbot/conversation.py` | 대화 세션 저장 (서버 메모리) |
| `chatbot/cli.py` | 터미널 대화 |
| `server/main.py` | FastAPI 앱, 웹 화면 서빙 |
| `server/routes.py` | `/api/chat`, `/api/chat/stream`, `/api/sessions`, `/api/models` |
| `server/schemas.py` | 요청/응답 형식 |
| `server/security.py` | 접속 비밀번호, 요청 수 제한 |
| `web/` | 채팅 화면 (`index.html`, `style.css`, `app.js`) |
| `training/dataset.py` | 학습 데이터(JSONL) 로드·검증 |
| `training/data_preprocessing.py` | 토크나이즈, 답변 부분만 학습하도록 라벨 마스킹 |
| `training/lora.py` | LoRA / 4bit 설정 |
| `training/train.py` | 학습 실행 |
| `training/evaluate.py` | 베이스 vs LoRA 답변 비교 |
| `notebooks/` | Colab 학습 / 서빙 노트북 |
| `configs/` | `model.yaml`(모델·프롬프트), `server.yaml`(보안·제한), `train.yaml`(학습) |
| `data/examples/` | 예시 학습 데이터, 평가 질문 |
| `tests/` | 단위 테스트 (API 키·GPU 없이 실행) |

## 1. 설치 (Windows, Python 3.11)

레포가 구글 드라이브 안에 있으므로 **가상환경은 드라이브 밖**에 만듭니다 (동기화로 느려지는 것 방지).

```powershell
py -3.11 -m venv C:\venvs\sol_chat
C:\venvs\sol_chat\Scripts\activate
cd "G:\내 드라이브\sol_chat"
pip install -r requirements.txt
copy .env.example .env
```

conda를 쓴다면 venv 대신 이렇게 해도 됩니다 (conda 환경은 기본적으로 사용자 폴더에 만들어져 드라이브 밖에 있습니다).
패키지는 `conda install` 이 아니라 pip 로 설치합니다.

```powershell
conda create -n sol_chat python=3.11 -y
conda activate sol_chat
cd "G:\내 드라이브\sol_chat"
pip install -r requirements.txt
copy .env.example .env
```

`.env` 를 열어서 채웁니다.

- `OPENROUTER_API_KEY`: https://openrouter.ai/keys 에서 발급
- `ACCESS_CODE`: 웹 화면 접속 비밀번호 (길고 추측하기 어렵게)

> OpenRouter 대시보드에서 키에 **사용 한도(credit limit)** 를 걸어두는 것을 추천합니다.
> 외부에 공개했을 때 혹시 모를 과금을 막아줍니다.

## 2. 터미널에서 대화 (Stage 1)

```powershell
python -m chatbot.cli
```

- `/reset` 새 대화, `/models` 모델 목록, `/model <이름>` 모델 변경, `exit` 종료
- 시스템 프롬프트, 모델, temperature 등은 `configs/model.yaml` 에서 바꿉니다.
- Gemini 외 모델도 OpenRouter 모델 ID만 바꾸면 됩니다 (https://openrouter.ai/models).

## 3. 웹 화면 (Stage 2)

```powershell
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 http://localhost:8000 → 비밀번호 입력 → 채팅.

- 답변이 실시간으로 흘러나오고(스트리밍), 왼쪽에서 새 대화·대화 전환·삭제가 됩니다.
- 대화는 서버 메모리에 저장되므로 **서버를 재시작하면 사라집니다.**
- 브라우저마다 대화가 분리되어 다른 사람의 대화는 보이지 않습니다.
- 제한 값은 `configs/server.yaml` 에서 바꿉니다 (분당 채팅 30회, 메시지 2000자 등).

### 외부에서 접속하기 (Cloudflare Tunnel)

공유기 설정 없이 무료로 HTTPS 공개 주소를 만듭니다.

1. cloudflared 설치: `winget install --id Cloudflare.cloudflared`
2. 서버를 켠 상태에서 다른 터미널에서:

   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```

3. 출력되는 `https://xxxx.trycloudflare.com` 주소로 휴대폰 등에서 접속합니다.

주의:
- 서버는 꼭 `--host 127.0.0.1` 로 띄우세요. 외부 접속은 터널로만 들어오게 해야 요청 제한이 정확히 동작합니다.
- 임시 터널 주소는 실행할 때마다 바뀝니다. 고정 주소가 필요하면 Cloudflare 계정 + 도메인으로 named tunnel 을 만드세요
  (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/).
- PC와 서버가 켜져 있는 동안만 접속할 수 있습니다.

## 4. LoRA 학습 (Stage 3, Colab)

1. Colab에서 `notebooks/train_lora_colab.ipynb` 를 엽니다
   (Drive에서 파일을 더블클릭 → Google Colaboratory로 열기).
2. `런타임 > 런타임 유형 변경 > T4 GPU` 선택 후 위에서부터 실행합니다.
   - Drive를 마운트해서 이 폴더를 그대로 쓰므로 업로드가 필요 없습니다.
   - 결과 어댑터는 `outputs/adapters/<run_name>/` 에 저장됩니다.
   - 세션이 끊기면 다시 실행하면 체크포인트에서 이어서 학습합니다.
3. 마지막 셀에서 베이스 모델과 LoRA 모델의 답변 비교가 나옵니다.

설정은 `configs/train.yaml` (베이스 모델, LoRA r/alpha, 학습률, 에포크 등).
학습 데이터 형식은 [`data/examples/README.md`](data/examples/README.md) 참고.

> 현재 베이스 모델(`Qwen/Qwen2.5-1.5B-Instruct`)과 예시 데이터는 **임시 제안안**입니다.
> PLAN.md의 "Phase 3 착수 전 결정 사항"(TBD-1~3)을 정하면서 바꾸면 됩니다.

### 학습한 모델과 웹에서 대화하기

두 가지 방법이 있습니다 (PLAN.md D-006).

**(a) Colab에서 서빙 (추천, 빠름)** — `notebooks/serve_vllm_colab.ipynb` 실행 → 마지막에 출력되는 내용을
PC의 `.env` 와 `configs/model.yaml` 에 붙여넣고 서버 재시작 → 웹 화면의 모델 선택에서 고르기.
Colab 세션이 살아 있는 동안만 동작합니다.

**(b) PC에서 직접 로드 (느림)** — PC에 `pip install -r requirements-train.txt` 후
`configs/model.yaml` 의 `my-lora-local` 항목 주석을 풀고 서버 재시작. GPU가 없으면 CPU로 돌아서 답변이 느립니다.

## 테스트

```powershell
python -m pytest -q
```

API 키나 GPU 없이 가짜 백엔드로 돌아갑니다. `bash scripts/verify.sh` 도 같은 테스트를 실행합니다.

## 개발 방식

이 레포는 계획 → 승인 → 태스크 → 실행 → 검증 → 보고 순서로 작업하는 개발 하네스를 포함합니다.
자세한 내용은 [`docs/harness.md`](docs/harness.md) 와 `AGENTS.md` 를 보세요.
