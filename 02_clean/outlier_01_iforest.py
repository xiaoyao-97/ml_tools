"""跑模型
from pyod.models.iforest import IForest
model = IForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0,
                bootstrap=False, n_jobs=-1, random_state=42)
model.fit(data)
y_train_scores = model.decision_scores_
print(y_train_scores[:10])
"""

"""feature importance
fe_if = model.feature_importances_
fe_if_df = pd.DataFrame(fe_if, columns=["score"], index=data.columns.to_list())
pd.set_option('display.max_rows', 100)
fe_if_df.sort_values("score",ascending = False)[:100]
# 画柱状图
top_f = fe_if_df.sort_values("score", ascending=False)[:20]
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
bars = plt.bar(top_f.index, top_f["score"])
plt.xticks(rotation=45, ha='right')
plt.xlabel("Column Name")
plt.ylabel("Score")
plt.title("Top 10 Columns by Score")
plt.show()
"""

"""sklearn
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100, random_state=1)
clf.fit(df)
scores = clf.decision_function(df)
df["score_if_01"] = scores
df['rank_if_01'] = df["score_if_01"].rank()
"""



