"""方法：
MI分数
RFE

The 4 practical ways of feature selection which yield best results are as follows: （https://www.kaggle.com/code/prashant111/comprehensive-guide-on-feature-selection）
SelectKBest
Recursive Feature Elimination
Correlation-matrix with heatmap
Random-Forest Importance
"""

"""MI
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def mi(df, cols, target):
    if df[target].dtype == 'object' or len(df[target].unique()) < 20:
        target_encoded = LabelEncoder().fit_transform(df[target])
        mi_values = mutual_info_classif(df[cols], target_encoded)
    else:
        mi_values = mutual_info_regression(df[cols], df[target])
    
    mi_df = pd.DataFrame(mi_values, index=cols, columns=['MI'])
    return mi_df

df = midf3.sort_values('MI', ascending = False)['MI']
plt.figure(figsize=(20, 12)) 
plt.bar(df.index, df.values)
plt.title('Bar Chart of Series')
plt.xlabel('Index')
plt.ylabel('Values')
plt.xticks(rotation=90)
plt.show()

"""

"""chi2 （分类变量之间）
# chi2统计量大,表示该特征与目标变量存在较强的相关性或依赖性。也就是说,保留该特征对预测目标变量是有帮助的。
# 对于连续变量，直接应用卡方检验并不合适，因为卡方检验假设输入变量是分类变量。
def features_chi2(df, cols, target):
    X = df[cols].astype(int)
    y = df[target]
    
    chi2_scores, p_values = chi2(X, y)
    
    results = pd.DataFrame({
        'Feature': cols,
        'Chi-squared Score': chi2_scores,
        'p-value': p_values
    })
    
    results = results.sort_values(by='Chi-squared Score', ascending=False).reset_index(drop=True)
    results['Rank'] = results.index + 1
    
    return results
"""

""" ANOVA F score
from sklearn.feature_selection import f_classif

def features_anova(df, cols, target):
    # 提取特征和目标变量
    X = df[cols]
    y = df[target]
    
    # 计算每个特征的F统计量和p值
    f_scores, p_values = f_classif(X, y)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Feature': cols,
        'F-score': f_scores,
        'p-value': p_values
    })
    
    # 按F统计量排序，并添加排名列
    results = results.sort_values(by='F-score', ascending=False).reset_index(drop=True)
    results['Rank'] = results.index + 1
    
    return results
"""





