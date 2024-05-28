# AR

from statsmodels.tsa.ar_model import AutoReg
np.random.seed(0)
data = np.random.randn(100)

model = AutoReg(data, lags=1)
model_fit = model.fit()

print(model_fit.summary())

# MA
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)
data = np.random.randn(100)

model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()

print(model_fit.summary())

# ARIMA
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)
data = np.random.randn(100).cumsum()

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())


# ARIMAX
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)
data = np.random.randn(100).cumsum()
exog = np.random.randn(100)

model = ARIMA(data, order=(1, 1, 1), exog=exog)
model_fit = model.fit()

print(model_fit.summary())

# SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(0)
data = np.random.randn(100).cumsum()

model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

print(model_fit.summary())

————————————————————————

from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(0)
data = np.random.randn(100).cumsum()
exog = np.random.randn(100)

model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=exog)
model_fit = model.fit()

print(model_fit.summary())




# 
