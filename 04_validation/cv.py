from sklearn.model_selection import cross_val_score
mse_scores = cross_val_score(rfr, X, y, scoring='neg_mean_squared_error', cv=5)

# 手动写：
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(clf, X, y, cv=kf)

# metric是我手动定义的：
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer

def metric(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

scorer = make_scorer(metric)

kf = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(clf, X, y, cv=kf, scoring=scorer)
print(cv_scores)

# scoring:
"""分类问题的评分指标
accuracy: 准确度，正确分类的样本比例。
precision: 精确率，真正例与预测为正例的样本的比例。
recall: 召回率，真正例与实际为正例的样本的比例。
f1: F1分数，精确率和召回率的调和平均。
roc_auc: 接收者操作特征曲线下的面积，衡量分类模型的性能。
average_precision: 预测正例的平均精确率。

回归问题的评分指标
neg_mean_squared_error: 均方误差的负值（因为scikit-learn在评分时越大越好，所以通常取负值）。
neg_mean_absolute_error: 平均绝对误差的负值。
r2: R²分数，表示模型对变异的解释程度。
neg_root_mean_squared_error: 均方根误差的负值。
neg_mean_squared_log_error: 对数均方误差的负值。

聚类问题的评分指标
adjusted_rand_score: 调整后的兰德指数。
homogeneity_score: 同质性得分。
"""

""" lgbm cv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# 假设已经定义了data和features0
X = data[data.type != 'test'][features0]
y = data[data.type != 'test']['yield']

kf = KFold(n_splits=10, shuffle=True, random_state=0)

pa = {
    'n_estimators': 265, 
    'num_leaves': 93, 
    'min_child_samples': 20,
    'learning_rate': 0.055, 
    'log_max_bin': 10, 
    'colsample_bytree': 0.88, 
    'reg_alpha': 0.00098, 
    'reg_lambda': 0.016,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbosity': -1
}

maes = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = lgb.LGBMRegressor(**pa)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae')
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)
    print(mae)

average_mae = np.mean(maes)
print(f'Average MAE: {average_mae:.4f}')
"""

"""通用模式
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(df))
losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    fold_loss = mean_squared_error(y_val, val_preds)
    losses.append(fold_loss)

    print(f"Fold {fold + 1} MSE: {fold_loss}")

# 计算总体OOF预测的均方误差
oof_loss = mean_squared_error(y, oof_predictions)
print(f"Overall OOF MSE: {oof_loss}")

"""
