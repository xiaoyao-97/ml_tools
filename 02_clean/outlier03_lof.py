"""sklearn
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit(df)
negative_outlier_scores = clf.negative_outlier_factor_  # Get outlier scores
df["score_lof_01"] = negative_outlier_scores
df['rank_lof_01'] = df["score_lof_01"].rank()
"""