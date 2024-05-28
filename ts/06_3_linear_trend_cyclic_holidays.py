"""找出假期
def rolling_mean_ratio(df,y_col,n):
    df[y_col+f"_rolmean_{n}"] = df[y_col].rolling(window=n, min_periods=1).mean()
    df["ratio"] = df[y_col]/df[y_col+f"_rolmean_{n}"]
    return df

def frequent_holidays(dfs, y_col, rolling_n, threshold):
    def rolling_mean_ratio(df,y_col,n):
        df[y_col+f"_rolmean_{n}"] = df[y_col].rolling(window=n, min_periods=1).mean()
        df["ratio"] = df[y_col]/df[y_col+f"_rolmean_{n}"]
        return df
    result = []
    for df in dfs:
        df = rolling_mean_ratio(df,y_col,rolling_n)
        result.append(df[df["ratio"]>threshold])
    result = pd.concat(result)
    return result["date"].apply(lambda x: (x.month,x.day)).value_counts()

dfs = []
for keys, group in enumerate(grouped):
    dfs.append(group[1])
holidays = frequent_holidays(dfs, "num_sold", 30, 1.3)
holidays[:17].index.to_list()
"""

"""傅里叶线性回归
def fourier_lr(ts,n):

    features = []
    
    for i in range(7):
        ts[str(i)] = ts.date.apply(lambda x:1 if x.dayofweek == i else 0)
        features.append(str(i))
    
    ts["doy"] = ts.date.dt.dayofyear
    
    ts["ndy"] = ts.date.apply(lambda x:366 if x.year%4==0 else 365)
    
    import math

    for i in range(1,n+1):
        ts["cos_"+str(i)] = np.cos(2*math.pi*i*ts.doy/ts.ndy)
        ts["sin_"+str(i)] = np.sin(2*math.pi*i*ts.doy/ts.ndy)
        features += ["cos_"+str(i),"sin_"+str(i)]
    
    ts["nod"] = (ts['date'] - ts['date'].iloc[0]).dt.days
    
    features.append("nod")
    return ts, features

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

for i in range(1,15):
    ts = ts0.copy()
    ts, features = fourier_lr(ts,i)
    X_train = ts[ts.date.dt.year<2017][features]
    X_test = ts[ts.date.dt.year==2017][features]
    y_train = ts[ts.date.dt.year<2017][target_col]
    y_test = ts[ts.date.dt.year==2017][target_col]
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train)
    results = model.fit()
    y_pred = results.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("i:",i, "Mean Squared Error on Test Set:", mse)

"""

"""趋势分解
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(ts1.set_index("date")[target_col], model='additive')  # 或者使用 model='multiplicative'
result.plot()

# 打印趋势、季节性和残差
print("Trend:")
print(result.trend)
print("\nSeasonal:")
print(result.seasonal)
print("\nResidual:")
print(result.resid)


另一个趋势分解：
from statsmodels.tsa.seasonal import STL

stl = STL(ts1.set_index("date")[target_col], seasonal=13)
result = stl.fit()

seasonal, trend, resid = result.seasonal, result.trend, result.resid
result.plot()
"""




