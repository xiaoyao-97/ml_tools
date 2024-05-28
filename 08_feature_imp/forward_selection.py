"""注意事项：
1. n_jobs = 1
2. 用什么模型就用这个模型来选参数，不然意义不大

"""

"""lgbm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# param = {'num_leaves': 86, 'max_depth': 12, 'learning_rate': 0.15861686681405443, 'n_estimators': 98, 'min_child_samples': 17, 'subsample': 0.525113456846555, 'colsample_bytree': 0.8604261275426714, 'reg_alpha': 0.3874638122628234, 'reg_lambda': 0.04413018526450343, 'verbosity':-1}
model = lgb.LGBMRegressor(num_leaves=86, 
                          max_depth=12, 
                          learning_rate=0.15861686681405443, 
                          n_estimators=98, 
                          min_child_samples=17, 
                          subsample=0.525113456846555,
                          colsample_bytree=0.8604261275426714, 
                          reg_alpha=0.3874638122628234,
                          reg_lambda=0.04413018526450343, 
                          verbosity=-1)
sfs1 = SFS(model, 
           k_features=10, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='neg_mean_squared_error',
           cv=10,
           n_jobs=1)

sfs1 = sfs1.fit(np.array(train[features1]), train['p'])

# 打印出features selected：
selected_features = [features1[idx] for idx in sfs1.k_feature_idx_]
print("Selected features:", selected_features)

print("Feature selection scores:", sfs1.k_score_)

# 训练结果：
print("Considered feature subsets:", sfs1.subsets_)

# 所有的score：
cv_results = sfs1.get_metric_dict()
pd.DataFrame(cv_results)
"""

"""rf
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
sfs2 = SFS(model, 
           k_features=15, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='neg_mean_squared_error',
           cv=5,
           n_jobs=1)

sfs2 = sfs2.fit(np.array(train[features1]), train['p'])
"""










