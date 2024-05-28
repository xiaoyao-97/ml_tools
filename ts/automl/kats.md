———————————————load data—————————————————————————

from kats.consts import TimeSeriesData

ts = TimeSeriesData(df)
multi_ts = TimeSeriesData(multi_ts_df)

ts = TimeSeriesData(timedf.time, value=df.value)
multi_ts = TimeSeriesData(time=df.time, value=df[['v1', 'v2']])

————————————————数据的操作——————————————————————————————
ts_1.extend(ts_2) 连接两个ts

ts..to_dataframe()

————————————————画图————————————————————————————————————
ts.plot(cols=['value'])
plt.show()

多变量：
multi_ts.plot(cols=['v1','v2'])
plt.show()


# —————————————————————不存在的AutoTS———————————————————————————————
from kats.models.auto_ts import AutoTS

model = AutoTS(
    forecast_length=10,  # Number of future time steps to forecast
    frequency='D',       # Frequency of the time series (e.g., 'D' for daily)
    holiday_country=None # Optional: specify the country for holidays
)

model.fit(df, time_col='time', value_col='value')

forecast = model.predict()
print(forecast)


# —————————————————————OutlierDetector————————————————————————————————————————
from kats.detectors.outlier import OutlierDetector

ts_outlierDetection = OutlierDetector(ts, 'additive') # call OutlierDetector
ts_outlierDetection.detector()

ts_outlierDetection.outliers[0]

ts_outliers_removed = ts_outlierDetection.remover(interpolate = False) # No interpolation
ts_outliers_interpolated = ts_outlierDetection.remover(interpolate = True) # With interpolation

fig, ax = plt.subplots(figsize=(20,8), nrows=1, ncols=2)

air_passengers_ts_outliers_removed.to_dataframe().plot(x = 'time',y = 'y_0', ax= ax[0])
ax[0].set_title("Outliers Removed : No interpolation")
air_passengers_ts_outliers_interpolated.to_dataframe().plot(x = 'time',y = 'y_0', ax= ax[1])
ax[1].set_title("Outliers Removed : With interpolation")
plt.show()

——————————————————训练单个模型———————————————————————————————————

from kats.models.prophet import ProphetModel, ProphetParams

params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

m = ProphetModel(ts, params)

m.fit()

fcst = m.predict(steps=30, freq="MS")

# 结果画图
m.plot()


# ————————————————————SARIMA—————————————————————————————————————
params = SARIMAParams(
    p = 2, 
    d=1, 
    q=1, 
    trend = 'ct', 
    seasonal_order=(1,0,1,12)
    )

m = SARIMAModel(data=air_passengers_ts, params=params)

m.fit()

fcst = m.predict(
    steps=30, 
    freq="MS",
    include_history=True
    )

m.plot()



# —————————————————————Prophet—————————————————————————————————

from kats.models.prophet import ProphetModel, ProphetParams


params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

m = ProphetModel(air_passengers_ts, params)

m.fit()

fcst = m.predict(steps=30, freq="MS")

m.plot()

# ——————————————————————Theta————————————————————————————————————
from kats.models.theta import ThetaModel, ThetaParams

params = ThetaParams(m=12)

m = ThetaModel(data=air_passengers_ts, params=params)

m.fit()

res = m.predict(steps=30, alpha=0.2)

m.plot()

# ————————————————————————Holt-Winters——————————————————————————————————

from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
warnings.simplefilter(action='ignore')


params = HoltWintersParams(
            trend="add",
            #damped=False,
            seasonal="mul",
            seasonal_periods=12,
        )
m = HoltWintersModel(
    data=air_passengers_ts, 
    params=params)

m.fit()
fcst = m.predict(steps=30, alpha = 0.1)
m.plot()


# ———————————————————————————ensemble—————————————————————————————————————
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,  # requires fbprophet be installed
    quadratic_model,
    sarima,
    theta,
)

model_params = EnsembleParams(
            [
                BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                BaseModelParams(
                    "sarima",
                    sarima.SARIMAParams(
                        p=2,
                        d=1,
                        q=1,
                        trend="ct",
                        seasonal_order=(1, 0, 1, 12),
                        enforce_invertibility=False,
                        enforce_stationarity=False,
                    ),
                ),
                BaseModelParams("prophet", prophet.ProphetParams()),  # requires fbprophet be installed
                BaseModelParams("linear", linear_model.LinearModelParams()),
                BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                BaseModelParams("theta", theta.ThetaParams(m=12)),
            ]
        )

KatsEnsembleParam = {
    "models": model_params,
    "aggregation": "median",
    "seasonality_length": 12,
    "decomposition_method": "multiplicative",
}

m = KatsEnsemble(
    data=air_passengers_ts, 
    params=KatsEnsembleParam
    )

m.fit()

fcst = m.predict(steps=30)

m.aggregate()

m.plot()




