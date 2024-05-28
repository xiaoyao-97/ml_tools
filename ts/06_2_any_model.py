"""分级数据
def ts_model_by_each_group(model, train, test, matrix_cols, y_col, date_col, feature_cols=[], val=False):
    result = []
    losses = []
    for keys, group in train.groupby(matrix_cols):
        condition = (test[matrix_cols].values == np.array(keys)).all(axis=1)
        test_ds = test[condition]

        train_features = [date_col, y_col] + feature_cols
        train_tmp = group[train_features]

        test_features = [date_col] + feature_cols
        if val:
            test_features.append(y_col)
        test_tmp = test_ds[test_features]

        model.fit(train_tmp.drop(columns=[y_col,date_col]), train_tmp[y_col])

        forecast = model.predict(test_tmp.drop(columns=[y_col,date_col], errors='ignore'))
        forecast = pd.DataFrame(forecast, columns=[y_col + '_predicted'])
        # print(forecast)
        for i, col in enumerate(matrix_cols):
            forecast[col] = keys[i]
        forecast[date_col] = test_tmp[date_col].values  

        for i, col in enumerate(matrix_cols):
            print(col, ":", keys[i])

        if val:
            rmse = np.sqrt(mean_squared_error(test_tmp[y_col], forecast[y_col + '_predicted']))
            print('RMSE:', rmse)
            print('R2 Score:', r2_score(test_tmp[y_col], forecast[y_col + '_predicted']))
            loss = rmse
            losses.append(loss)

        # print(forecast)
        result.append(forecast)

    result = pd.concat(result)
    final_result = test.merge(result, on=[date_col] + matrix_cols, how="left")
    return final_result, losses
"""


