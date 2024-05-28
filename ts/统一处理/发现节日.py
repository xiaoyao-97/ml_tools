# 用p_value找节日
import pandas as pd
import statsmodels.api as sm

def find_significant_dates(df, date_col, col, features):
    results = []

    df['date_without_year'] = df[date_col].dt.strftime('%m-%d')

    unique_dates = df['date_without_year'].unique()

    for date in unique_dates:
        df['special_date'] = (df['date_without_year'] == date).astype(int)

        X = df[features + ['special_date']]
        y = df[col]

        combined = pd.concat([X, y], axis=1).dropna()

        X = combined[features + ['special_date']]
        y = combined[col]

        model = sm.OLS(y, X).fit()

        p_value = model.pvalues['special_date']
        t_stat = model.tvalues['special_date']

        results.append({'date': date, 'p_value': p_value, 't_stat': t_stat})

    results_df = pd.DataFrame(results)

    return results_df


# 对多个target ts找节日

import pandas as pd
import statsmodels.api as sm

def find_significant_dates_multi(df, date_col, cols, features):
    # 创建一个列表来存储结果
    results = []

    # 提取所有不包含年份的日期
    df['date_without_year'] = df[date_col].dt.strftime('%m-%d')

    # 获取所有不包含年份的唯一日期
    unique_dates = df['date_without_year'].unique()

    # 遍历所有的日期
    for date in unique_dates:
        # 创建一个新的特征变量
        df['special_date'] = (df['date_without_year'] == date).astype(int)

        # 初始化p值和t统计量的字典
        p_values = {}
        t_stats = {}

        for target in cols:
            # 选择特征变量和目标变量
            X = df[features + ['special_date']]
            y = df[target]

            # 将X和y合并成一个DataFrame，以便同时处理缺失值
            combined = pd.concat([X, y], axis=1).dropna()

            # 分开特征变量和目标变量
            X = combined[features + ['special_date']]
            y = combined[target]

            # 添加常数项
            X = sm.add_constant(X)

            # 进行线性回归
            model = sm.OLS(y, X).fit()

            # 获取special_date的p值和t统计量
            p_values[target] = model.pvalues['special_date']
            t_stats[target] = model.tvalues['special_date']

        # 将结果添加到列表中
        result = {'date': date}
        for target in cols:
            result[f'p_value_{target}'] = p_values[target]
            result[f't_stat_{target}'] = t_stats[target]
        results.append(result)

    # 将列表转换为DataFrame
    results_df = pd.DataFrame(results)

    return results_df




