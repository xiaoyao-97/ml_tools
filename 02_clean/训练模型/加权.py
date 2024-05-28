# lgbm
import lightgbm as lgb
import numpy as np
import pandas as pd

X = pd.DataFrame(np.random.rand(100, 10))  # 特征数据
y = np.random.rand(100)  # 目标变量
weights = np.random.rand(100)  # 样本权重

lgb_train = lgb.Dataset(X, label=y, weight=weights)

params = {
    'objective': 'regression',
    'metric': 'l2'
}

model = lgb.train(params, lgb_train, num_boost_round=100)


# xgb
import xgboost as xgb
import numpy as np
import pandas as pd

X = pd.DataFrame(np.random.rand(100, 10))  # 特征数据
y = np.random.rand(100)  # 目标变量
weights = np.random.rand(100)  # 样本权重

dtrain = xgb.DMatrix(X, label=y, weight=weights)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

model = xgb.train(params, dtrain, num_boost_round=100)



# rf
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=10, random_state=42)
weights = np.random.rand(100)  # 样本权重

model = RandomForestRegressor()

model.fit(X, y, sample_weight=weights)





