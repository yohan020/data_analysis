import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



plt.style.use('seaborn-v0_8')
sns.set(font_scale=2.5)

df_train = pd.read_csv('./input/train.csv')
df_train.drop('Id', axis=1, inplace=True)

df_train_num = df_train.select_dtypes(include=['float64','int64'])

def split_dataset(dataset, test_ratio=0.3):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(df_train)

label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)

rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])
rf.fit(x=train_ds)

tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

inspector = rf.make_inspector()

#print(f"Available variable importances:")
#for importance in inspector.variable_importances().keys():
#    print("\t",importance)

plt.figure(figsize=(12,4))

# variable_importance_metric = "NUM_AS_ROOT"
# variable_importance = inspector.variable_importances()[variable_importance_metric]

# feature_names = [vi[0].name for vi in variable_importance]
# feature_importances = [vi[1] for vi in variable_importance]

# feature_ranks = range(len(feature_names))

# bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
# plt.yticks(feature_ranks, feature_names)
# plt.gca().invert_yaxis()

# for importance, patch in zip(feature_importances, bar.patches):
#     plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

# plt.xlabel(variable_importance_metric)
# plt.title("NUM AS ROOT of the class 1 vs the others")
# plt.tight_layout()
# plt.show()

df_test = pd.read_csv('./input/test.csv')
ids = df_test.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, task = tfdf.keras.Task.REGRESSION)
# preds = rf.predict(test_ds)
# output = pd.DataFrame({'Id':ids, 'SalePrice': preds.squeeze()})

df_submit = pd.read_csv('./input/sample_submission.csv')
df_submit['SalePrice'] = rf.predict(test_ds)
df_submit.to_csv('./input/submit.csv', index=False)
