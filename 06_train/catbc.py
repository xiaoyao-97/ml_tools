# 基本train-val

from catboost import CatBoostClassifier, Pool
import pandas as pd
import matplotlib.pyplot as plt

train_data = Pool(X_train, y_train)
val_data = Pool(X_val, y_val)

model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.01,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    od_type='Iter',
    od_wait=100,
    logging_level='Verbose'
)

model.fit(train_data, eval_set=val_data, use_best_model=True, verbose=1000)

train_metrics = pd.DataFrame(model.eval_metrics(train_data, ['AUC'])['AUC'])
val_metrics = pd.DataFrame(model.eval_metrics(val_data, ['AUC'])['AUC'])

train_metrics.columns = ['train_auc']
val_metrics.columns = ['val_auc']

metrics_df = pd.concat([train_metrics, val_metrics], axis=1)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df.index.values, metrics_df['train_auc'].values, label='Train Loss')
plt.plot(metrics_df.index.values, metrics_df['val_auc'].values, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#




