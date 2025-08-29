import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

plt.style.use('seaborn-v0_8')
sns.set(font_scale=2.5)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None) 

import missingno as msno
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')
df_submit = pd.read_csv('./input/sample_submission.csv')

#print(df_train.info())
print('-----------\n',df_train[['LotFrontage','MasVnrArea','GarageYrBlt']].describe())
