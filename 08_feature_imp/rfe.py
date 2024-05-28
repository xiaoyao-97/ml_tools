"""基本代码
from sklearn.feature_selection import RFE (RFECV)


X, y = data1[data1.type=='train'][features1].fillna(0), data1[data1.type=='train']['p']

model = RandomForestRegressor()

selector = RFE(estimator=model, n_features_to_select=10, step=1)
selector = selector.fit(X, y)

# 输出选定的特征
print("Selected features (True = selected, False = not selected):")
print(selector.support_)

# 输出特征的排名
print("Feature ranking (1 = most important, higher numbers = less important):")
print(selector.ranking_)

# 检查并打印最终模型的参数
print("Parameters of the fitted model:")
print(selector.estimator_.get_params())

# 打印模型的系数（如果适用）
if hasattr(selector.estimator_, 'coef_'):
    print("Coefficients of the fitted model:")
    print(selector.estimator_.coef_)
else:
    print("No coefficients available for this model type.")

# features排名
feature_names = features1
feature_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': selector.ranking_
})
feature_ranking.sort_values('Ranking')

sorted_features = feature_ranking.sort_values('Ranking')['Feature'].tolist()


"""







