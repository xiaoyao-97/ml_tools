"""
变量区分：target 和 covariate，都可以是多变量
    past_covariates是在预测的时候不知道的
    future_covariates是在预测时可以知道的
"""


"""create a series
dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')

data1 = np.random.randn(len(dates))

series1 = TimeSeries.from_times_and_values(dates, data1, freq='D')

"""

"""multivariate_series
my_multivariate_series = concatenate([series1, series2], axis=1)
"""

"""series的功能：
series.pd_dataframe() 把series变成df
series.values() 不如.pd_dataframe().values
"""

