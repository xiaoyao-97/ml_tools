import statsmodels.api as sm

# 添加常数项以拟合截距
df = sm.add_constant(df)

model = sm.OLS(y, df)
results = model.fit()

# 输出回归结果摘要
print(results.summary())

# 获取特定变量的系数（beta）
beta_X1 = results.params['X1']

# 获取特定变量的t统计量
tstat_X1 = results.tvalues['X1']

# 获取特定变量的p值
pvalue_X1 = results.pvalues['X1']

# 获取特定变量的标准差
stderr_X1 = results.bse['X1']