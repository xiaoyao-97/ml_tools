# 均方误差 (Mean Squared Error, MSE):
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

# 平均绝对误差 (Mean Absolute Error, MAE):
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

# 平方根均方误差 (Root Mean Squared Error, RMSE):
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)

# R-squared (R²) 得分:
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

# 调和平均 F1 得分:
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

# 精确率 (Precision), 召回率 (Recall), F1-score:
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# 混淆矩阵 (Confusion Matrix):
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# 分类准确率 (Classification Accuracy):
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# ROCAUC
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred_prob)