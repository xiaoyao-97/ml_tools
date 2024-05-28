"""基础
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
traval = pd.concat([tra,val])
scaler = StandardScaler()
traval[features] = scaler.fit_transform(traval[features])
tra2 = traval[:1800]
val2 = traval[1800:]
for a in range(1,100):
    alpha = a/10000
    ridge = Ridge(alpha=alpha)
    ridge.fit(tra2[features], tra2['l1'])
    y_pred_ridge = ridge.predict(val2[features])
    mse_ridge = mse(val2['l1'], y_pred_ridge)
    print(alpha, f'Ridge MSE: {np.sqrt(mse_ridge)}')
"""


"""optuna+cv
import optuna
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-3, 10.0)  # 正则化项的系数
    solver = trial.suggest_categorical('solver', ['svd', 'cholesky', 'lsqr', 'sag'])
    
    model = Ridge(alpha=alpha, solver=solver)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return np.mean(score)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print('Best parameters:', best_params)

best_ridge = Ridge(**best_params)
best_ridge.fit(X_train, y_train)

y_pred = best_ridge.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
"""

