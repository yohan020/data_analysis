import pandas as pd
from datasets import Dataset

df_train = pd.read_csv('llm-classification-finetuning/train.csv')

# 새로운 데이터 저장을 위한 리스트
new_df_train = []

for _, row in df_train.iterrows():
    text_input = f"prompt: {row['prompt']} response_a: {row['response_a']} response_b: {row['response_b']}"
    if row['winner_model_a'] == 1:
        label = 0
    elif row['winner_model_b'] == 1:
        label = 1
    elif row['winner_tie'] == 1:
        label = 2
    else:
        continue
    new_df_train.append({'text': text_input, 'label':label})

# 리스트를 hugging face dataset으로 변환
train_dataset = Dataset.from_pandas(pd.DataFrame(new_df_train))

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

model_name = "microsoft/deberta-v3-small"

tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=3)

from transformers import TrainingArguments, Trainer

# 텍스트 데이터를 토큰화 하는 함수
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

# 데이터셋에 토큰화 적용
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    per_device_train_batch_size=1,
)

# Trainer 인스턴스 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

# 훈련 시작
trainer.train()

import torch

df_test = pd.read_csv('llm-classification-finetuning/test.csv')

prob_a_list = []
prob_b_list = []
prob_tie_list = []

# 모델을 평가 모드로 전환 (추론시 필수)
model.eval()

with torch.no_grad():
    for _, row in df_test.iterrows():
        # 모델 입력 텍스트 준비
        text_input = f"prompt: {row['prompt']} response_a: {row['response_a']} response_b: {row['response_b']}"
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True).to(model.device)


        # 모델의 로짓 출력 얻기
        logits = model(**inputs).logits

        # Softmax를 적용하여 로짓을 확률로 변환
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

        # 각 클래스의 확률을 리스트에 추가
        prob_a_list.append(probs[0])
        prob_b_list.append(probs[1])
        prob_tie_list.append(probs[2])

submission_df = pd.DataFrame({
    'id': df_test['id'],
    'winner_model_a': prob_a_list,
    'winner_model_b': prob_b_list,
    'winner_tie': prob_tie_list
})

submission_df.to_csv("submission.csv", index=False)