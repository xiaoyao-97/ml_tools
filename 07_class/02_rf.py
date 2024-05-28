from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.metrics import roc_auc_score

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # 获取正类别的预测概率

# 计算ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC:", roc_auc)









# gridsearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Test score with best parameters: {test_score}")


"""params_grid
params_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}"""





# 可以打印进度的grid search：
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# 创建RandomForestClassifier实例
rf = RandomForestClassifier(n_jobs=-1)
# 设置超参数网格
param_grid = {
    'n_estimators': [150],
    'max_depth': list(range(35, 45,1))
}
# 用于记录最优模型和最高分数
best_model = None
best_score = 0
best_params = None
# 遍历所有参数组合
for i, params in enumerate(ParameterGrid(param_grid)):
    print(f"Testing parameter set {i+1}/{len(ParameterGrid(param_grid))}: {params}")
    # 更新模型参数
    rf.set_params(**params)
    # 训练模型
    rf.fit(X_train, y_train)
    # 使用模型进行预测
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    # 计算ROC AUC分数
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC for current parameters: {roc_auc:.4f}\n")
    # 比较和更新最高分数及最优模型
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = clone(rf)
        best_params = params
# 输出最佳参数和ROC AUC分数
print("Best Parameters:", best_params)
print("Best ROC AUC:", best_score)

