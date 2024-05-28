# Voting

from sklearn.ensemble import VotingClassifier

estimator = [] 
estimator.append(('LR',LogisticRegression(solver ='lbfgs',multi_class ='multinomial',max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier())) 

hard_voting = VotingClassifier(estimators = estimator, voting ='hard') 
hard_voting.fit(X_train, y_train) 
y_pred = hard_voting.predict(X_test)   

"""软投票（Soft Voting）：

在软投票中，每个模型的预测结果都会有一个概率值（即属于每个类别的概率），而不只是简单的类别标签预测。软投票通过平均每个模型的预测概率来进行投票决策。
通常情况下，对于分类任务，软投票可以提供比硬投票更好的性能，因为它利用了更多的信息（概率值）进行决策。
加权投票（Weighted Voting）：

在加权投票中，每个模型的预测结果根据其性能或可信度赋予不同的权重，这些权重可以手动指定或根据模型的准确性动态确定。
加权投票可以使性能较高的模型在投票中拥有更大的影响力，从而提高整体投票结果的准确性。"""





# Stacking
base_models = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]
meta_model = LogisticRegression(random_state=42)

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)

#但是通常，我们需要知道训练的中间结果，而不仅仅是一个简单的最终结果，因此就需要手动写函数来进行运算

# regression

def single_model_regression_cv(model, X_train, X_test, y_train, cv=5, seed=42, verbose=1, metric="r2"):
    from sklearn.model_selection import KFold
    from sklearn.base import clone
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    np.random.seed(seed)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    
    # Choose the metric
    if metric == "r2":
        scorer = r2_score
    elif metric == "mse":
        scorer = mean_squared_error
    elif metric == "mae":
        scorer = mean_absolute_error
    else:
        raise ValueError("Unsupported metric. Use 'r2', 'mse', or 'mae'.")
    
    # Initialize lists to store fold results
    metrics = []
    train_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Clone and fit the model
        cloned_model = clone(model)
        cloned_model.fit(X_train_fold, y_train_fold)
        
        # Predictions
        val_pred = cloned_model.predict(X_val_fold)
        train_predictions[val_idx] = val_pred
        
        # Evaluate metric
        fold_metric = scorer(y_val_fold, val_pred)
        metrics.append(fold_metric)
        
        if verbose:
            print(f"Fold {fold} {metric}: {fold_metric:.4f}")
    
    # Predict on the test set
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    
    # Print average metric
    if verbose:
        print(f"Average {metric}: {np.mean(metrics):.4f} (+/- {np.std(metrics):.4f})")
    
    # Results dictionary
    results = {
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "cv_metrics": metrics,
        "cv_mean": np.mean(metrics),
        "cv_std": np.std(metrics)
    }
    
    return results



