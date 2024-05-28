import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def weighted_cv(model, X_train, y_train, folds=3, random_state=1):
    tscv = TimeSeriesSplit(n_splits=folds)
    scores = []
    weights = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        model.random_state = random_state
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_test_fold)
        score = mean_squared_error(y_test_fold, predictions)
        scores.append(score)
        
        # Weight recent folds more
        weight = len(y_train) - test_index[0]
        weights.append(weight)
        
        print(f"Fold {fold+1}: MSE = {score}")

    # Calculate weighted average of scores
    weighted_scores = np.average(scores, weights=weights)
    print(f"Weighted average MSE: {weighted_scores}")

# Example usage:
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
# weighted_cv(model, X_train, y_train)


"""通用模型
def ts_model_cv(model, data, matrix_cols, y_col, date_col, feature_cols=[], n_splits=5):
    data = data.sort_values(by=date_col)
    
    unique_dates = data[date_col].unique()
    n_dates = len(unique_dates)
    fold_size = n_dates // n_splits

    results = []
    losses = []
    for i in range(n_splits-1):
        if i < n_splits - 2:
            train_dates = unique_dates[:fold_size * (i + 1)]
            test_dates = unique_dates[fold_size * (i + 1): fold_size * (i + 2)]
        else:
            train_dates = unique_dates[:fold_size * (i + 1)]
            test_dates = unique_dates[fold_size * (i + 1):]
        
        train = data[data[date_col].isin(train_dates)]
        test = data[data[date_col].isin(test_dates)]
        # print(train.head(),test.head())

        res, loss = ts_model_by_each_group(model, train, test, matrix_cols, y_col, date_col, feature_cols, val=True)
        results.append(res)
        losses.append(sum(loss)/len(loss))
    print(losses)
    print("avg_loss is: ", sum(losses)/n_splits)
    result = pd.concat(results)
    final_result = test.merge(result[[date_col, y_col+"_predicted"] + matrix_cols], on=[date_col] + matrix_cols, how="left")
    return final_result, losses

model = LinearRegression()
ts_model_cv(model, data[data.type == "train"], matrix_cols, y_col, date_col,feature_cols)
"""

"""


"""





