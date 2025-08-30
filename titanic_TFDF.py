import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

# -------------------------
# 1. 데이터 불러오기
# -------------------------
df_train = pd.read_csv('./titanic/train.csv')
df_test = pd.read_csv('./titanic/test.csv')

# -------------------------
# 2. 불필요한 ID 제거
# -------------------------
df_train.drop('PassengerId', axis=1, inplace=True)
ids = df_test['PassengerId']
df_test.drop('PassengerId', axis=1, inplace=True)
# Cabin 열 제거
df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)


# -------------------------
# 3. 범주형 데이터 전처리
# -------------------------
# 성별: male=0, female=1
df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})

# Embarked: C=0, Q=1, S=2
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# NaN 값은 -1로 채움
df_train.fillna(-1, inplace=True)
df_test.fillna(-1, inplace=True)

# -------------------------
# 4. TFDF 데이터셋 변환
# -------------------------
label = 'Survived'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label=label, task=tfdf.keras.Task.CLASSIFICATION)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, task=tfdf.keras.Task.CLASSIFICATION)

# -------------------------
# 5. 모델 생성 및 학습
# -------------------------
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.CLASSIFICATION)
model.compile(metrics=["accuracy"])
model.fit(train_ds)

# -------------------------
# 6. 예측
# -------------------------
preds = model.predict(test_ds)
output = pd.DataFrame({'PassengerId': ids, 'Survived': preds.squeeze().astype(int)})
output.to_csv('./input/titanic_submit.csv', index=False)

print("모델 학습 및 예측 완료!")
