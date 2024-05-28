import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns             
import scipy.stats as stats       

train = pd.read_csv("train.csv")
train["type"] = "train"
test = pd.read_csv("test.csv")
test["type"] = "test"
data = pd.concat([train,test])
data.reset_index(inplace = True)

# 机器学习
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression      
from sklearn.ensemble import RandomForestClassifier    


df.describe()


import warnings

# 关闭警告
warnings.filterwarnings('ignore')

# 你的代码块
# df_tmp["log_sales"] = y_pred

# 恢复默认的警告设置
warnings.filterwarnings('default')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)

# 有nan的行
def get_rows_with_nan(df, cols):
    return df[df[cols].isna().any(axis=1)]

