import shap

# 创建SHAP解释器
xgb_explainer = shap.TreeExplainer(
    model, X_train, feature_names=X_train.columns.tolist()
)
shap_values = xgb_explainer.shap_values(X_train, y_train)

"""重要：给shap importance的df
def get_shap_importance(X, features, shap_values):
    shap_importance = pd.DataFrame({
        'feature': features,
        'importance': np.abs(shap_values).mean(axis=0)
    })

    shap_importance = shap_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

    shap_importance['rank'] = shap_importance['importance'].rank(ascending=False, method='min').astype(int)

    return shap_importance
"""

# 重要性的bar图
shap.summary_plot(shap_values, X, plot_type="bar")

# 决策图（树状图）：
shap.decision_plot(lgb_explainer.expected_value, shap_values, X)

# SHAP 总结图 (SHAP Summary Plot)
shap.summary_plot(shap_values, X)

shap_values_xgb = shap_values_xgb[:, :-1]
pd.DataFrame(shap_values_xgb, columns=X_train.columns.tolist()).head()

# 每个feature和shap值的散点图
for feature_name in features2:
    shap.dependence_plot(feature_name, shap_values, X)

# 绘制依赖图，展示特征1与特征2的交互
primary_feature = "hp"
interaction_feature = "b"
shap.dependence_plot(primary_feature, shap_values, X, interaction_index=interaction_feature)

'''两个特征的shap的散点图
feature1 = "hp"
feature2 = "b"

# 提取这两个特征的 SHAP 值
shap_values_feature1 = shap_values[:, X.columns.get_loc(feature1)]
shap_values_feature2 = shap_values[:, X.columns.get_loc(feature2)]

# 绘制交互散点图
plt.figure(figsize=(8, 6))
plt.scatter(shap_values_feature1, shap_values_feature2, alpha=0.5)
plt.xlabel(f"SHAP values for {feature1}")
plt.ylabel(f"SHAP values for {feature2}")
plt.title(f"SHAP Interaction Scatter Plot: {feature1} vs {feature2}")
plt.grid(True)
plt.show()
'''

# 可视化第一个预测的SHAP值
shap.initjs()
shap.force_plot(lgb_explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# SHAP 水位图 (SHAP Waterfall Plot)
# 注意：这个图通常用于展示单个样本的 SHAP 值
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=lgb_explainer.expected_value, data=X.iloc[0]))


# 创建一个依赖性图，显示特征和SHAP值之间的关系
for i in range(min(5, X.shape[1])):  # 仅绘制前5个特征的依赖图
    shap.dependence_plot(i, shap_values[1], X_test, feature_names=[f"Feature {i}" for i in range(X.shape[1])])





"""eli5
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt

# 生成示例数据集
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df['target'], test_size=0.2, random_state=42)

# 训练LightGBM回归模型
model = lgb.LGBMRegressor(objective='regression', 
                          learning_rate=0.1, 
                          num_leaves=31, 
                          feature_fraction=0.9, 
                          bagging_fraction=0.8, 
                          bagging_freq=5, 
                          n_estimators=1000)
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          eval_metric='rmse', 
          )

# 使用PermutationImportance来解释模型
perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)

# 获取特征重要性和标准差
weights = eli5.explain_weights_df(perm, feature_names=feature_names)
weights = weights[['feature', 'weight', 'std']]

# 打印特征重要性和标准差
print(weights)

# 画柱状图，包含误差区间
plt.figure(figsize=(10, 6))
plt.bar(weights['feature'], weights['weight'], yerr=weights['std'], capsize=5)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance with Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""


