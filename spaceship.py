import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('./spaceship-titanic/train.csv')
df_test = pd.read_csv('./spaceship-titanic/test.csv')
df_submit = pd.read_csv('.//spaceship-titanic/sample_submission.csv')

df_train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
df_train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df_train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)

label = 'Transported'

df_train[label] = df_train[label].astype(int)
df_train['VIP'] = df_train['VIP'].astype(int)
df_train['CryoSleep'] = df_train['CryoSleep'].astype(int)

df_train[['Deck','Cabin_num','Side']] = df_train['Cabin'].str.split('/', expand=True)
try:
    df_train.drop('Cabin', axis=1, inplace=True)
except:
    print("Field does not exists")

def split_dataset(dataset, test_ratio=0.2):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]
train_ds_pd, valid_ds_np = split_dataset(df_train)
print("{} examples in training, {} examples in testing".format(len(train_ds_pd), len(valid_ds_np)))

# pandas.DataFrame -> tf.data.Dataset으로 변환해줌, 텐서플로운에 맞는 데이터 형식이 필요함
# label = label은 어느 라벨이 정답인지 지정하는 것
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_np, label=label)

rf = tfdf.keras.RandomForestModel() # 트리 갯수는 디폴트로 300개가 생성됨
rf.compile(metrics=["accuracy"]) # 모델을 컴파일하며, 평가지표로 정확도를 설정함
rf.fit(x=train_ds) # rf를 train_ds로 학습시킴

inspector = rf.make_inspector()

print(inspector.variable_importances()['NUM_AS_ROOT']) # 여기서 점수가 높은 특징은 중요도가 높은거

submission_id = df_test.PassengerId # df_test에서 승객 아이디 칼럼 추출

# 데이터 전처리
df_test[['VIP', 'CryoSleep']] = df_test[['VIP','CryoSleep']].fillna(value=0) 
df_test[['Deck','Cabin_num','Side']] = df_test['Cabin'].str.split('/', expand=True)
df_test.drop('Cabin', axis=1, inplace=True)
df_test['VIP'] = df_test['VIP'].astype(int)
df_test['CryoSleep'] = df_test['CryoSleep'].astype(int)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test) # 전처리된 데이터 형식 변환
predictions = rf.predict(test_ds) # 랜덤 포레스트 모델을 사용해서 예측값 생성
n_predictions = (predictions > 0.5).astype(bool) # 출력이 확률이므로 이를 0.5를 기준으로 해서 True False 분류
output = pd.DataFrame({'PassengerId' : submission_id,
                       'Transported':n_predictions.squeeze()}) # 스퀴즈는 불필요한 차원을 제거해서 1차원 형태로 정리 [0.43], [0.23] -> [0.43, 0.23]으로 바꿈

df_submit[label] = n_predictions
df_submit.to_csv('./submission.csv', index=False)

