# df display
from pandas_profiling import ProfileReport
profile = ProfileReport(df)
# debug: profile = ProfileReport(df, config_file='')
profile


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



from dabl import plot
plot(train.drop("SalePrice",axis = 1), train["SalePrice"])



#SNS

"""sns————————————————————————————————————————————————————————————————
sns.histplot
sns.kdeplot
sns.jointplot
    kind="regg" 就可以画回归图

sns关系

# 数值与数值的关系：
sns.scatterplot
sns.regplot

# 分类与分类的关系：
sns.heatmap

# 每个分类的数值图
sns.countplot(【pandas.Series】,label="Count") 
sns.barplot

# 分类与数值的交互
plt.figure(figsize=(10,10))
sns.boxplot(x=分类变量, y=数值变量, hue=另一个分类变量, data=data)
或者：sns.violinplot(x=分类变量, y=数值变量, hue=另一个分类变量, data=data, split=True, inner="quart")
或者：sns.swarmplot(x=分类变量, y=数值变量, hue=另一个分类变量, data=data)
plt.xticks(rotation=90) #把名字旋转90度

# 时间序列
sns.lineplot
"""

"""pandas单变量————————————————————————————————————————————————————————————————————
数值变量：
个数统计的bar图：【pandas.Series】.value_counts().head(10).plot.bar()

分类变量
个数统计的bar图：【pandas.Series】.value_counts().sort_index().plot.bar()
sns.countplot(【pandas.Series】,label="Count") 
或者.line()
或者.area()

【pandas.Series】.plot.hist()
"""

"""pandas双变量————————————————————————————————————————————————————————————————————
数值和数值
df.plot(kind = "scatter", x=col1, y = col2)

分类变量的多列的bar
df.plot.bar(stacked=True)
你也可以自己分类：
new_df = df.groupby(col1,...,).mean()[[col2,...]]
new_df.plot.bar(stacked=True)
pd.crosstab(data.col1,data.col2d,margins=True).style.background_gradient(cmap='summer_r')
"""


"""pair plot
df = 选择一些列
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# correlation
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


"""



