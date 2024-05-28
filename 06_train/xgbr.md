# 基本的train-val

import xgboost as xgb

train_data = xgb.DMatrix(X_train, label=y_train)
val_data = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.9
}

evals_result = {}
model = xgb.train(
    params,
    train_data,
    num_boost_round=10000,
    evals=[(train_data, 'train'), (val_data, 'val')],
    early_stopping_rounds=100,
    evals_result=evals_result,
    verbose_eval=1000
)

train_metric_name = list(evals_result['train'].keys())[0]
val_metric_name = list(evals_result['val'].keys())[0]

train_metrics = pd.DataFrame(evals_result['train'])
val_metrics = pd.DataFrame(evals_result['val'])

train_metrics.columns = [f'train_{col}' for col in train_metrics.columns]
val_metrics.columns = [f'val_{col}' for col in val_metrics.columns]

metrics_df = pd.concat([train_metrics, val_metrics], axis=1)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df.index.values, metrics_df[f'train_{train_metric_name}'].values, label='Train Loss')
plt.plot(metrics_df.index.values, metrics_df[f'val_{val_metric_name}'].values, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()




