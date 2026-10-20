# Report: 베이스라인 평가(CRITIC) 지적 사항 수정

- id: 20261020-critic-fixes
- date: 2026-10-20
- plan: docs/reports/20261020-baseline-CRITIC.md 의 "7. Claude에 전달할 수정 범위" (B-001~B-006)

## Summary

평가서의 6건(B-001~B-006)을 먼저 회귀 테스트로 재현해서 6건 모두 실제로 발생하는 것을 확인했고, 전부 수정했다.
낮은 우선순위로 분류된 요청 제한 기록 정리도 함께 반영했다. 새 기능은 추가하지 않았다.

## Changes

- B-001 `chatbot/generation.py`: 모델별 초기화 잠금. 같은 모델의 첫 요청이 겹쳐도 백엔드는 한 번만 생성, 다른 모델은 서로 기다리지 않음, 실패하면 다음 요청에서 재시도.
- B-002 `chatbot/backends/local_backend.py`: 생성 스레드 예외를 기록하고 `streamer.end()` 로 스트림을 반드시 종료, 소비자 쪽에서 `BackendError` 로 변환. 다음 글자 대기 시간 120초 제한. 일반 생성도 예외를 `BackendError` 로 변환.
- B-003 `chatbot/conversation.py`, `configs/server.yaml`(`sessions.max_turns_per_session: 100`): 세션 보관 메시지 수 상한. 넘으면 오래된 턴부터 user/assistant 쌍 단위로 삭제.
- B-004 `web/app.js`: `done`/`error` 없이 끝난 스트림을 오류로 처리하고, 서버에 저장된 대화로 화면을 다시 그림.
- B-005 `server/security.py`: 비밀번호 실패 한도를 넘은 IP는 비밀번호 비교 전에 429. `RateLimiter.is_limited()` 추가.
- B-006 `chatbot/conversation.py`, `server/routes.py`: 세션 단위 "생성 중" 잠금(생성 시작~저장). 같은 대화의 동시 요청은 409. 정상/오류/연결 끊김 모두에서 해제하고, 해제 못 한 잠금은 5분 뒤 무시.
- 낮은 우선순위 `server/security.py`: 요청이 끊긴 IP의 기록을 주기적으로 삭제.
- `tests/test_robustness.py`: 위 항목의 회귀 테스트 9개.

## Verification

- verdict: PASS
- 수정 전: 새 테스트 9개 중 6개 실패 (B-001, B-002, B-003, B-005, B-006, 기록 정리) → 수정 후 전체 통과
- `bash scripts/verify.sh` -> pytest 69 passed, verify: OK
- B-004: 브라우저(Playwright)에서 `done` 없이 끊기는 스트림을 가로채 넣었을 때 오류 표시 + 서버 기록으로 화면 복원 + 입력창 다시 활성화 확인. 기존 화면 흐름(로그인, 스트리밍, 새 대화, 대화 다시 열기, 휴대폰 메뉴)도 다시 확인.
- B-002: 작은 테스트 모델로 로컬 백엔드 일반/스트리밍 생성이 정상 동작하는 것도 다시 확인.

## Deviations from plan

- B-006 은 "순차 처리" 대신 "409 거절"을 선택. 대기시키면 요청이 길게 묶이고, 화면은 이미 한 탭에서 생성 중 전송을 막고 있어 두 번째 탭에 이유를 바로 알려주는 쪽이 단순하다.

## Deferred / Remaining risks

- 스트림이 시작되기도 전에 연결이 끊기면 세션 잠금이 최대 5분 남을 수 있다 (`busy_timeout_seconds`). 그동안 같은 대화에 보내면 409, 새 대화는 영향 없음.

## Context Index

- docs/reports/20261020-baseline-CRITIC.md — contains: 외부 평가 원문 — use_for: 지적 근거
- tests/test_robustness.py — contains: B-001~B-006 회귀 테스트 — use_for: 재발 확인
