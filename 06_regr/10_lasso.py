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

"""GridSearch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd

class ProgressDisplay:
    def __init__(self, param_grid, n_splits):
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.total_runs = n_splits * np.prod([len(v) for v in param_grid.values()])
        self.current_run = 0

    def update(self, params, fold, score):
        self.current_run += 1
        progress = (self.current_run / self.total_runs) * 100
        print(f"Params: {params} | Fold: {fold + 1}/{self.n_splits} | Loss: {score:.4f} | Progress: {progress:.2f}%")

def custom_scorer(display, estimator, X, y):
    y_pred = estimator.predict(X)
    loss = np.mean((y - y_pred) ** 2)  # Mean squared error
    fold = display.current_run % display.n_splits
    params = estimator.get_params()
    display.update(params, fold, loss)
    return -loss  # Negative MSE because GridSearchCV maximizes the score

model = Lasso(random_state=1)
param_grid = {
    'alpha': [0.01, 0.1, 1, 10],  # Regularization strength
    'max_iter': [1000, 5000, 10000],  # Maximum number of iterations
    'selection': ['cyclic', 'random']  # Algorithm for coordinate descent
}
cv = 5
random_state = 1
kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

progress_display = ProgressDisplay(param_grid, cv)

grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=lambda estimator, X, y: custom_scorer(progress_display, estimator, X, y))
grid_search.fit(train.drop(columns=[target]), train[target])  # Assuming 'train' and 'target' are defined

results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score', 'std_test_score']]
print(results)"""

"""optuna+cv
import optuna
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-3, 1.0)  # 正则化项的系数
    
    model = Lasso(alpha=alpha, random_state=42)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return np.mean(score)

X_train, X_test, y_train, y_test = train[features], test[features], train[target], test[target]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # 可以调整n_trials来控制搜索的迭代次数

best_params = study.best_params
print('Best parameters:', best_params)

best_lasso = Lasso(**best_params)
best_lasso.fit(X_train, y_train)

y_pred = best_lasso.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
"""




