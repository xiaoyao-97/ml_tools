"""基本代码：
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, y_train = train[features], train[target]

X_test, y_test = test[features], test[target]

model = ElasticNet(alpha=1.0, l1_ratio=0.5)  # alpha是正则化项的强度，l1_ratio是L1正则化的比例（与L2相比）

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean Squared Error: {rmse}')
"""

"""grid search
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# 假设train和test已经被定义并且包含了features和target
X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = ElasticNet()

param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # 正则化强度
    'l1_ratio': np.linspace(0.1, 1, 10)  # L1和L2的比例
}

search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, return_train_score=True)

search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)

cv_results = search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(params, 'has mean score:', -mean_score)

for i in range(len(cv_results['params'])):
    fold_scores = cv_results['split0_test_score'][i], cv_results['split1_test_score'][i], cv_results['split2_test_score'][i], cv_results['split3_test_score'][i], cv_results['split4_test_score'][i]
    print(f"Scores for {cv_results['params'][i]}: {[-score for score in fold_scores]}")

y_pred = search.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)
"""

"""optuna+cv
import optuna
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_regression


X_train, X_test, y_train, y_test = train[features], test[features], train[target], test[target]

def objective(trial):
    alpha = trial.suggest_float('alpha', 0.2, 0.35)  # 正则化项的系数
    l1_ratio = trial.suggest_float('l1_ratio', 0.8, 1.0)  # L1正则化的比例
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return np.mean(score)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 可以调整n_trials来控制搜索的迭代次数

best_params = study.best_params
print('Best parameters:', best_params)

best_model = ElasticNet(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
"""

"""hyperopt+cv
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

space = {
    'alpha': hp.uniform('alpha', 0.25, 0.32),  # 正则化项的系数
    'l1_ratio': hp.uniform('l1_ratio', 0.98, 1.0)  # L1正则化的比例
}

def objective(params):
    model = ElasticNet(**params)
    score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return {'loss': -np.mean(score), 'status': STATUS_OK}

trials = Trials()

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=150, trials=trials)
print('Best parameters:', best)

best_model = ElasticNet(**best)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
"""




