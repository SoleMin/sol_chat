# 예시 데이터

- `sample.jsonl`: 학습 동작 확인용 예시 대화 33개. 말투는 "친근한 반말 + 마지막 줄에 `📌 한 줄 요약:`" 으로 통일해서,
  학습 후 이 형식이 나오는지로 학습 효과를 눈으로 확인할 수 있게 했다.
- `eval_questions.txt`: 학습 전후 비교 질문 (`python -m training.evaluate`).

형식 (한 줄에 대화 하나):

```json
{"messages": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
```

- `system` 메시지는 생략 가능. 생략하면 `configs/train.yaml` 의 `data.system_prompt` (없으면 `configs/model.yaml` 의 시스템 프롬프트)가 들어간다.
- 대화는 user 로 시작해서 assistant 로 끝나야 하고, 둘이 번갈아 나와야 한다.
- 내 데이터는 `data/raw/` 에 두고 (git 제외) `configs/train.yaml` 의 `train_file` 을 바꾼다.
