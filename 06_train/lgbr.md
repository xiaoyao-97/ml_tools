# 基本train-val

import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

evals_result = {}
callbacks = [
    lgb.record_evaluation(evals_result),
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=1000) # num_rounds to print result
]
model = lgb.train(
    params,
    train_data,
    num_boost_round=10000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=callbacks
)

train_metrics = pd.DataFrame(evals_result['train'])
val_metrics = pd.DataFrame(evals_result['val'])

train_metrics.columns = [f'train_{col}' for col in train_metrics.columns]
val_metrics.columns = [f'val_{col}' for col in val_metrics.columns]

metrics_df = pd.concat([train_metrics, val_metrics], axis=1)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df.index.values, metrics_df['train_rmse'].values, label='Train Loss')
plt.plot(metrics_df.index.values, metrics_df['val_rmse'].values, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# CV




