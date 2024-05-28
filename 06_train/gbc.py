# 基本 train-val
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

gbc = GradientBoostingClassifier(
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    max_features=0.9,
    validation_fraction=0.1,
    n_iter_no_change=100,
    tol=1e-4,
    verbose=1
)

gbc.fit(X_train, y_train)

train_auc = []
val_auc = []

for train_pred, val_pred in zip(gbc.staged_predict_proba(X_train), gbc.staged_predict_proba(X_val)):
    train_auc.append(roc_auc_score(y_train, train_pred[:, 1]))
    val_auc.append(roc_auc_score(y_val, val_pred[:, 1]))

train_metrics = pd.DataFrame(train_auc, columns=['train_auc'])
val_metrics = pd.DataFrame(val_auc, columns=['val_auc'])
metrics_df = pd.concat([train_metrics, val_metrics], axis=1)

plt.figure(figsize=(10, 6))
plt.plot(metrics_df.index.values, metrics_df['train_auc'].values, label='Train AUC')
plt.plot(metrics_df.index.values, metrics_df['val_auc'].values, label='Validation AUC')
plt.xlabel('Iterations')
plt.ylabel('AUC')
plt.title('Training and Validation AUC')
plt.legend()
plt.show()


# 






