"""主要就是一个TimeSeriesPredictor

TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-m4-hourly",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)

方法：
fit
fit_summary
model_names（模型名称）
leaderboard
load（储存现有模型）
feature_importance
evaluate
predict
plot

TimeSeriesPredictor.evaluate(
    data: TimeSeriesDataFrame | DataFrame | Path | str, 
    model: str | None = None, 
    metrics: str | TimeSeriesScorer | List[str | TimeSeriesScorer] | None = None, 
    display: bool = False, use_cache: bool = True) → Dict[str, float]

"""

"""fit and predict
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)

train_data.head()

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    time_limit=600,
)

predictions = predictor.predict(train_data)
predictions.head()

predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)

predictor.leaderboard(test_data)
"""

"""如果有外生变量
直接把feature放在df的列中就行。
data = pd.DataFrame({
    "timestamp": pd.date_range(start="2021-01-01", periods=100, freq="H"),
    "feature1": range(100),
    "feature2": range(100, 200),
    "feature3": range(200, 300),
    "target": range(300, 400)
})

train_data = TimeSeriesDataFrame(data, index_column="timestamp")
"""

# 不支持multivariate

"""单个时间序列的函数
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def ts_aotogluon(data, tar_col, date_col, pred_len, feature_cols = []):
    df = data.reset_index()[[tar_col, date_col]+feature_cols].rename(columns = {tar_col:'target'})
    df['id_col'] = 'id_col'
    train_data = TimeSeriesDataFrame.from_data_frame(
        df,
        id_column = 'id_col',
        timestamp_column=date_col
    )
    
    train_data.head()
    
    predictor = TimeSeriesPredictor(
        prediction_length=pred_len,
        path="path01",
        target='target',
        eval_metric="MASE",
    )
    
    predictor.fit(
        train_data,
        time_limit=10,
    )
    
    predictions = predictor.predict(train_data)
    predictions = predictions.reset_index()[['timestamp','mean']].rename(columns = {'timestamp':date_col, 'mean':tar_col})
    return predictions
"""

"""多个时间序列，对melted df
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
tar_col = 'sales'
date_col = 'date'
id_column = 'item_id'
def each_aotogluon(data, tar_col, date_col, id_column, pred_len, feature_cols = []):
    train_data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column = id_column,
        timestamp_column = date_col
    )
    
    # print(train_data.head())
    
    predictor = TimeSeriesPredictor(
        prediction_length = pred_len,
        path = "path02",
        target = tar_col,
        eval_metric = "MASE",
    )
    
    predictor.fit(
        train_data,
        time_limit = 10,
    )
    
    predictions = predictor.predict(train_data)
    predictions = predictions.reset_index()[[id_column, "timestamp", "mean"]].rename(columns = {"timestamp":date_col, "mean":tar_col})
    return predictions
"""





