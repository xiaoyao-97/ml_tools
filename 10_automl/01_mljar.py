from supervised.automl import AutoML
automl = AutoML(
    algorithms = ['Random Forest', 'LightGBM', 'Xgboost', 'CatBoost'],
    mode="Compete", 
    total_time_limit=300,
)

automl.fit(
    train[features], 
    train[target]
)