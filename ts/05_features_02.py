"""rolling window趋势项
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 假设 df 是你的时间序列数据框，包含一个时间索引和一个值列
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum()
})
df.set_index('date', inplace=True)

# 计算时间差（以天为单位）
df['time_diff'] = (df.index - df.index[0]).days

# 定义一个函数来计算滑动窗口的线性回归 beta
def rolling_ewm_linear_regression(df, window, halflife):
    # 创建一个存储 beta 的空列
    df['beta'] = np.nan

    # 计算权重
    def calculate_weights(window, halflife):
        alpha = 1 - np.exp(np.log(0.5) / halflife)
        weights = np.exp(-alpha * np.arange(window))
        return weights / weights.sum()

    # 滑动窗口
    for i in range(1, len(df) + 1):
        window_size = min(i, window)  # 当前窗口大小
        
        # 获取当前窗口的数据
        y = df['value'].iloc[i-window_size:i]
        X = df['time_diff'].iloc[i-window_size:i]
        
        # 增加一个常数项
        X = sm.add_constant(X)
        
        # 指数加权
        weights = calculate_weights(window_size, halflife)
        
        # 线性回归
        model = sm.WLS(y, X, weights=weights).fit()
        df['beta'].iloc[i-1] = model.params[1]  # 获取 time_diff 的系数
    
    return df

# 设置窗口大小和半衰期
window_size = 30
halflife = 15

# 计算滑动窗口的线性回归 beta
df = rolling_ewm_linear_regression(df, window_size, halflife)

# 显示前几行数据
print(df.head(40))
"""

"""rolling variance
import pandas as pd
import numpy as np

# 假设 df 是你的时间序列数据框，包含一个时间索引和一个值列
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum()
})
df.set_index('date', inplace=True)

# 定义一个函数来计算滚动窗口的方差
def rolling_variance_before(df, window):
    # 创建一个存储方差的空列
    df['var'] = df['value'].rolling(window=window, min_periods=1).var()
    return df

# 设置窗口大小
window_size = 20

# 计算滚动窗口的方差（当前时间之前的值）
df = rolling_variance_before(df, window_size)

# 显示前几行数据
print(df.head(40))
"""

"""rolling variance + centered
import pandas as pd
import numpy as np

# 假设 df 是你的时间序列数据框，包含一个时间索引和一个值列
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum()
})
df.set_index('date', inplace=True)

# 定义一个函数来计算滚动窗口的方差
def rolling_variance_centered(df, window):
    # 创建一个存储方差的空列
    df['var'] = df['value'].rolling(window=window, center=True, min_periods=1).var()
    return df

# 设置窗口大小
window_size = 10

# 计算滚动窗口的方差（当前时间前后的值）
df = rolling_variance_centered(df, window_size)

# 显示前几行数据
print(df.head(50))"""

"""rw+lr+res+std
import pandas as pd
import numpy as np
import statsmodels.api as sm

def rolling_linear_regression_residual_sqrt_var(df, window):
    df['residual_sqrt_var'] = np.nan

    half_window = window // 2

    for i in range(len(df)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(df), i + half_window + 1)
        
        y = df['value'].iloc[start_idx:end_idx]
        X = df['time_diff'].iloc[start_idx:end_idx]
        
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        df.loc[df.index[i], 'res_std'] = np.sqrt(np.var(residuals))
    
    return df

# 示例数据
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum()
})
df.set_index('date', inplace=True)

# 计算时间差（以天为单位）
df['time_diff'] = (df.index - df.index[0]).days

# 设置窗口大小
window_size = 30

# 计算滚动窗口的线性回归残差方差的平方根
df = rolling_linear_regression_residual_sqrt_var(df, window_size)

# 显示前几行数据
print(df.head(40))
"""




import pandas as pd
import numpy as np

# 假设df是你的时间序列数据框，包含一个时间索引和一个值列
df = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'value': np.random.randn(100).cumsum()
})
df.set_index('date', inplace=True)

# 时间特征
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['weekday'] = df.index.weekday
df['dayofyear'] = df.index.dayofyear
df['weekofyear'] = df.index.isocalendar().week

# 滞后特征
df['lag_1'] = df['value'].shift(1)
df['lag_7'] = df['value'].shift(7)
df['lag_30'] = df['value'].shift(30)

# 滚动窗口特征
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_max_7'] = df['value'].rolling(window=7).max()
df['rolling_min_7'] = df['value'].rolling(window=7).min()

# 指数加权移动平均
df['ewm_mean_7'] = df['value'].ewm(span=7).mean()

# 变化率特征
df['daily_growth'] = df['value'].pct_change()
df['weekly_growth'] = df['value'].pct_change(periods=7)

# 显示前几行数据
print(df.head())



