"""
n_jobs=-1,
random_state=1,
objective= "regression"
boosting_type='gbdt', 
'metric': 'rmse',

n_estimators=100,
learning_rate=0.1 (0.001-0.1)
num_leaves=31 (8-128)
max_depth=-1 (3-15)

feature_fraction
bagging_fraction
bagging_freq
subsample_for_bin=200000, 
min_split_gain=0.0, 

lambda_l1=0.0, 
lambda_l2=0.0, 
"""

# optuna
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = your_data, your_target  # Replace with your data and target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 1,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': trial.suggest_int('num_leaves', 20, 40),
        'max_depth': -1,
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'subsample_for_bin': 200000,
        'min_split_gain': 0.0,
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
    }
    
    gbm = lgb.LGBMRegressor(**param)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse')
    y_pred = gbm.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 平行图
optuna.visualization.plot_parallel_coordinate(study)

# 参数重要性
optuna.visualization.plot_param_importances(study)

# loss的分布
optuna.visualization.plot_edf(study)


#————————————————————————————hyperopt代码————————————————————————————————————————————————————————————
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X, y = your_data, your_target  # Replace with your actual data and target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def objective(params):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 1,
        'n_estimators': int(params['n_estimators']),
        'learning_rate': params['learning_rate'],
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']),
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': int(params['bagging_freq']),
        'subsample_for_bin': 200000,
        'min_split_gain': 0.0,
        'lambda_l1': params['lambda_l1'],
        'lambda_l2': params['lambda_l2'],
    }
    
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=10, verbose=False)
    y_pred = gbm.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return {'loss': rmse, 'status': STATUS_OK}

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'num_leaves': hp.quniform('num_leaves', 20, 40, 1),
    'max_depth': hp.quniform('max_depth', -1, 50, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    'bagging_freq': hp.quniform('bagging_freq', 0, 10, 1),
    'lambda_l1': hp.uniform('lambda_l1', 0, 1.0),
    'lambda_l2': hp.uniform('lambda_l2', 0, 1.0),
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print('Best hyperparameters:', best)



"""optuna + cv
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

X, y = data.data, data.target

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }

    model = lgb.LGBMRegressor(**param)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    return np.mean(rmse_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:', study.best_trial.params)
"""

"""optuna+cv version2:
import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

X = train[features]
y = train[target]

def objective(trial, X, y):
    FOLDS = 3
    # Define the parameter space dynamically within the objective function
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000),
        # "lambda_l1": trial.suggest_float("lambda_l1", 0, 100),
        # "lambda_l2": trial.suggest_float("lambda_l2", 0, 100),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95),
        'verbose': -1 
    }
    
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=1)
    cv_scores = np.empty(FOLDS)
    
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
        )
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # RMSE
        cv_scores[idx] = rmse
        print(f"Fold {idx+1}: RMSE = {rmse:.4f}")
    return np.mean(cv_scores)

# Example usage
study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=30)
"""

"""hyperopt+cv
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold

# Sample data
X, y = data[data.type!='test'][features], data[data.type!='test']['yield']  # Replace with your actual data and target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def objective(params):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 1,
        'verbosity': -1,
        'n_estimators': int(params['n_estimators']),
        'learning_rate': params['learning_rate'],
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']),
        # 'feature_fraction': params['feature_fraction'],
        # 'bagging_fraction': params['bagging_fraction'],
        # 'bagging_freq': int(params['bagging_freq']),
        # 'subsample_for_bin': 200000,
        # 'min_split_gain': 0.0,
        # 'lambda_l1': params['lambda_l1'],
        # 'lambda_l2': params['lambda_l2'],
    }
    
    gbm = lgb.LGBMRegressor(**params)
    # gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae')
    # y_pred = gbm.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # return {'loss': mae, 'status': STATUS_OK}
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=2)
    scores = cross_val_score(gbm, X, y, cv=cv_strategy, scoring=scorer)
    mae = -scores.mean()  # 因为make_scorer使用负值进行优化
    return {'loss': mae, 'status': STATUS_OK}

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 150, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.15),
    'num_leaves': hp.quniform('num_leaves', 17, 35, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    # 'feature_fraction': hp.uniform('feature_fraction', 0.5, 1.0),
    # 'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1.0),
    # 'bagging_freq': hp.quniform('bagging_freq', 0, 10, 1),
    # 'lambda_l1': hp.uniform('lambda_l1', 0, 1.0),
    # 'lambda_l2': hp.uniform('lambda_l2', 0, 1.0),
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print('Best hyperparameters:', best)"""



