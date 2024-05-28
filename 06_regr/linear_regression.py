import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设你已经有了X和y
# 这里我将生成一些模拟数据作为示例
np.random.seed(0)
X = np.random.rand(100, 2)  # 100个样本，2个特征
y = np.random.rand(100)  # 100个目标值

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 为训练集添加截距
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# 创建模型
model = sm.OLS(y_train, X_train_sm)

# 拟合模型
results = model.fit()

# 使用测试集进行预测
y_pred = results.predict(X_test_sm)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse)
