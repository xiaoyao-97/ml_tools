# Grubb test
import numpy as np
import pandas as pd
from scipy import stats

def grubbs_test(x):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        x = np.array(x)  # Convert input to NumPy array
    else:
        raise ValueError("Input must be a list, NumPy array, or pandas Series")

    n = len(x)
    mean_x = np.mean(x)
    sd_x = np.std(x)
    numerator = max(abs(x - mean_x))
    g_calculated = numerator / sd_x
    print("Grubbs Calculated Value:", g_calculated)
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
    print("Grubbs Critical Value:", g_critical)

    if g_critical > g_calculated:
        print("From grubbs_test we observe that calculated value is lesser than critical value, "
              "Accept null hypothesis and conclude that there are no outliers\n")
    else:
        print("From grubbs_test we observe that calculated value is greater than critical value, "
              "Reject null hypothesis and conclude that there are outliers\n")


# Z score
def Zscore_outlier(df):
    m = df.mean(axis=0)
    sd = df.std(axis=0)
    out = {} 

    for column in df.columns:
        z = (df[column] - m[column]) / sd[column]  # Calculate the Z-score for each entry in the column
        outliers = df[column][np.abs(z) > 5].tolist()  # Identify outliers
        if outliers:
            out[column] = outliers  # Store outliers if any

    print("Outliers:", out)

# robust Z-score
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

def ZRscore_outlier(df):
    out = {}

    for column in df.columns:
        med = np.median(df[column])
        ma = median_abs_deviation(df[column])

        zr = 0.6745 * (df[column] - med) / ma if ma != 0 else 0 

        outliers = df[column][np.abs(zr) > 5].tolist()
        if outliers:
            out[column] = outliers

    print("Outliers:", out)



# Winsorization
def Winsorization_outliers(df):
    out = []
    q1 = np.percentile(df , 1)
    q3 = np.percentile(df , 99)
    for i in df:
        if i > q3 or i < q1:
            out.append(i)
    print("Outliers:",out)

# Isolation forest
from sklearn.ensemble import IsolationForest
df_iso = df.copy()
model=IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.01), max_features=10)
model.fit(df_iso)
scores=model.decision_function(df_iso)
anomaly=model.predict(df_iso)
df_iso['scores']=scores
df_iso['anomaly']=anomaly
anomaly = df_iso.loc[df_iso['anomaly']==-1]
anomaly_index = list(anomaly.index)
print('Total number of outliers is:', len(anomaly))
df_out5 = df_iso.drop(anomaly_index, axis = 0).reset_index(drop=True)