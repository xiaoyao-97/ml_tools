import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score

matrix_cols = ['store', 'product']
y_col = "number_sold"
date_col = "Date"

def prophet_each_ts(train, test, matrix_cols, y_col, date_col, val=False):
    result = []

    for keys, group in train.groupby(matrix_cols):
        # Prepare test dataset based on 'val' flag
        if val:
            # Filtering 'test' dataframe to match the groups in 'train'
            condition = (test[matrix_cols].values == np.array(keys)).all(axis=1)
            test_ds = test[condition]
        else:
            # If not validation, use all unique dates from 'test'
            test_ds = test[[date_col]].drop_duplicates()
        
        # Preparing the data for Prophet
        train_tmp = group[[date_col, y_col]].rename(columns={date_col: 'ds', y_col: 'y'})
        if val:
            test_tmp = test_ds[[date_col, y_col]].rename(columns={date_col: 'ds', y_col: 'y'})
        else:
            test_tmp = test_ds.rename(columns={date_col: 'ds'})
        
        # Creating and fitting the Prophet model
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(train_tmp)
        
        # Making predictions
        forecast = model.predict(test_tmp)[["ds", "yhat"]]
        forecast = forecast.rename(columns={'ds': date_col, 'yhat': y_col + '_predicted'})
        
        # Assigning the group keys to the forecast DataFrame
        for i, col in enumerate(matrix_cols):
            forecast[col] = keys[i]
            print(col, ":", keys[i])

        # Calculating metrics if validation is true
        if val:
            merged_forecast = test[[date_col, y_col] + matrix_cols].merge(forecast, on=[date_col] + matrix_cols, how='inner')
            print('RMSE:', mean_squared_error(merged_forecast[y_col], merged_forecast[y_col + '_predicted'], squared=False))
            print('R2 Score:', r2_score(merged_forecast[y_col], merged_forecast[y_col + '_predicted']))
        
        print(forecast)
        result.append(forecast)

    # Concatenating all forecasts and merging with the original 'test' dataset
    result = pd.concat(result)
    final_result = test.merge(result, on=[date_col] + matrix_cols, how="left")
    return final_result

"""基本的例子
import pandas as pd
from prophet import Prophet

# 创建一个包含日期和目标变量的数据框
df = pd.DataFrame({
    'ds': pd.date_range(start='2022-01-01', periods=100),
    'y': (range(100)) + np.random.normal(scale=10, size=100)  # 模拟数据
})

df['holiday'] = 0
df.loc[df['ds'].dt.weekday > 5, 'holiday'] = 1 
df['promo'] = 0.5  
df['price'] = df['ds'].apply(lambda x: 20 if x.month in [6, 7, 8] else 30)  # 季节性价格变化

additional_regressors = ['holiday', 'promo', 'price']

m = Prophet()

for regressor in additional_regressors:
    m.add_regressor(regressor)

m.fit(df)

future = m.make_future_dataframe(periods=30)
future['holiday'] = 0
future.loc[future['ds'].dt.weekday > 5, 'holiday'] = 1
future['promo'] = 0.5
future['price'] = future['ds'].apply(lambda x: 20 if x.month in [6, 7, 8] else 30)

forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
"""


