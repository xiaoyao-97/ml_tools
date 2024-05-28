# permutation importance
def calculate_permutation_importance(model, X_val, y_val, feature_list, n_repeats=30, random_state=0):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        n_jobs=-1  # 使用所有可用的CPU核心
    )
    
    importance_data = {
        "features": feature_list,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }
    
    df_importance = pd.DataFrame(importance_data)
    
    df_importance.sort_values(by="importance_mean", ascending=False, inplace=True)
    return df_importance

# 画图
def plot_permutation_importance(df_importance):
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 8))
    barplot = plt.bar(
        df_importance['features'], 
        df_importance['importance_mean'], 
        yerr=df_importance['importance_std'], 
        capsize=5, 
        color='skyblue',  
        error_kw={'capthick': 2, 'elinewidth': 2}  
    )

    plt.title('Feature Permutation Importance with Standard Deviation')
    plt.xlabel('Features')
    plt.ylabel('Permutation Importance')

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


"""任何模型
决策树（Decision Trees）
DecisionTreeClassifier
DecisionTreeRegressor
随机森林（Random Forests）
RandomForestClassifier
RandomForestRegressor
梯度提升树（Gradient Boosting Trees）
GradientBoostingClassifier
GradientBoostingRegressor
极端随机树（Extra Trees）
ExtraTreesClassifier
ExtraTreesRegressor
线性模型
线性模型提供的系数可以用作特征重要性的一个指标，特别是在特征已经被标准化后，系数的大小可以直接表示特征的影响力。

线性回归
LinearRegression
逻辑回归
LogisticRegression
岭回归（Ridge Regression）
Ridge
Lasso回归
Lasso
Elastic Net
ElasticNet
集成模型
一些集成模型也支持特征重要性的评估。

AdaBoost
AdaBoostClassifier
AdaBoostRegressor
Bagging模型
BaggingClassifier
BaggingRegressor
"""
def calculate_built_in_feature_importance(model, X, y, feature_names):
    model.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if model.coef_.ndim == 1 else np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame()
    
    df_importance = pd.DataFrame({
        'features': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    print(df_importance)
    plt.bar(df_importance_tree['features'],df_importance_tree['importance'])
    plt.show()
    return df_importance

tree_model = LogisticRegression()
df_importance_tree = calculate_built_in_feature_importance(tree_model, X, y, feature_names)

