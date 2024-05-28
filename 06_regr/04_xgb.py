import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"训练集均方误差: {train_mse}")
print(f"测试集均方误差: {test_mse}")

"""optuna+cv
import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

X_train, y_train = train[features].values, train[target].values

# 定义目标函数
def objective(trial):
    # 为XGBoost定义超参数空间
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        # 'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
        # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1.0)
    }

    # 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train_kf, y_train_kf, eval_set=[(X_val_kf, y_val_kf)], early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_val_kf)
        rmse = mean_squared_error(y_val_kf, preds)
        scores.append(rmse)
    
    return np.mean(scores)

# 创建一个 Optuna 的 study 对象，并进行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 打印最优的参数和结果
print('Best trial:', study.best_trial.params)
"""






