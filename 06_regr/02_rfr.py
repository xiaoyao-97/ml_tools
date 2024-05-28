"""
n_estimators: 树的数量。更多的树可以提高模型的稳定性和性能，但也会增加计算成本。
max_depth: 树的最大深度。限制深度可以帮助防止模型过拟合，但如果深度太小，可能会导致欠拟合。
min_samples_split: 分裂内部节点所需的最小样本数。较大的值可防止过拟合，但过大可能导致欠拟合。
min_samples_leaf: 叶节点所需的最小样本数。这个参数同样影响模型的拟合程度，较大的值可以增加模型的泛化能力。
max_features: 寻找最佳分割时要考虑的特征数量。它决定了在每次分裂时用多少个特征来考虑。通常选项包括“auto”，“sqrt”，“log2”或特定数字。
bootstrap: 是否在构建树时使用bootstrap样本。这意味着在训练每棵树时是否使用随机抽样的方法。
max_leaf_nodes: 以最佳优先方式生长的树的最大叶子节点数。这可以帮助控制树的复杂度，防止过拟合。
"""

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': list(range(500,800,100)),  # 可以选择更多的值
    'max_depth': [18, 20, 22],  # None意味着没有限制
    'min_samples_split': [10,15,20],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': [0.5, 'sqrt', 'log2'],
    # 'bootstrap': [True, False],
    # 'max_leaf_nodes': [None, 10, 20, 30]  # None意味着没有限制
}

grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=6, verbose=1, n_jobs=-1)

grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)

# 设置不同的随机状态
rfr = RandomForestRegressor()

param_grid = {
    'n_estimators': list(range(500, 800, 100)),
    'max_depth': [18, 20, 22],
    'min_samples_split': [10, 15, 20]
}

# 设置不同的随机状态
random_states = [42, 50, 60]

# 存储每次运行的最佳分数
scores = []

for state in random_states:
    # 创建交叉验证对象，设置随机状态
    cv = KFold(n_splits=6, shuffle=True, random_state=state)
    grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters for random state {state}:", grid_search.best_params_)
    print(f"Best score for random state {state}:", -grid_search.best_score_)
    scores.append(-grid_search.best_score_)

# 计算平均分数
average_score = np.mean(scores)
print("Average Best Score:", average_score)


"""调参：hyperopt
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 假设 train_X, train_y, val_X, val_y 已经定义

def objective(params):
    model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_leaf=int(params['min_samples_leaf']),
        min_samples_split=int(params['min_samples_split'])
    )
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    score = mean_squared_error(val_y, pred)
    return {'loss': score, 'status': STATUS_OK}

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 250, 1),
    'max_depth': hp.quniform('max_depth', 20, 45, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 3, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 3, 1)
}

trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=350, trials=trials)

print("Best parameters:")
for param in best:
    print(param, ":", int(best[param]))
print("Best score:", trials.best_trial['result']['loss'])
"""


"""调参：optuna
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 假设 train_X, train_y, val_X, val_y 已经定义

def objective(trial):
    est = trial.suggest_int('n_estimators', 100, 500)
    md = trial.suggest_int('max_depth', 5, 20)
    msl = trial.suggest_int('min_samples_leaf', 1, 5)
    mss = trial.suggest_int('min_samples_split', 2, 6)
    model = RandomForestRegressor(n_estimators=est, max_depth=md, min_samples_leaf=msl, min_samples_split=mss)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    score = mean_squared_error(val_y, pred)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)
"""