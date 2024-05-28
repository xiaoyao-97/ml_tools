
# 分类变量与target的交互，统计数据量
result_df = pd.crosstab(df[col], df[target])
pd.set_option('display.max_rows', 100)
result_df