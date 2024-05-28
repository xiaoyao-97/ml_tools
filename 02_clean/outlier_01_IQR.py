# interquartile range (IQR)
""" 看看box图，了解超出的部分
def plot_boxplot(data):
    # 确保数据为 Pandas Series，以便使用quantile等方法
    if isinstance(data, list) or isinstance(data, np.ndarray):
        data = pd.Series(data)

    # 创建箱线图
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=data)

    # 计算边界点
    q1 = data.quantile(0.25)
    median = data.quantile(0.5)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    whisker_low = q1 - 1.5 * iqr
    whisker_high = q3 + 1.5 * iqr
    min_val = data.min()
    max_val = data.max()

    print(f'最小值: {min_val:.2f}')
    print(f'第一四分位数 (Q1): {q1:.2f}')
    print(f'中位数: {median:.2f}')
    print(f'第三四分位数 (Q3): {q3:.2f}')
    print(f'最大值: {max_val:.2f}')


    # 分位数的均匀分布，相对于整个数据集
    print("\n超出IQR下方的分位数临界点:")
    if min_val < whisker_low:
        # 确定超出IQR下方的数据百分比
        lower_data = data[data <= whisker_low]
        lower_percentiles = np.linspace(0, 1, 11)
        for p in lower_percentiles:
            value_at_percentile = lower_data.quantile(p)
            print(f'{data[data <= value_at_percentile].count() / len(data) * 100:.1f}%: {value_at_percentile:.2f}')

    print("\n超出IQR上方的分位数临界点:")
    if max_val > whisker_high:
        # 确定超出IQR上方的数据百分比
        upper_data = data[data >= whisker_high]
        upper_percentiles = np.linspace(0, 1, 11)
        for p in upper_percentiles:
            value_at_percentile = upper_data.quantile(p)
            print(f'{data[data >= value_at_percentile].count() / len(data) * 100:.1f}%: {value_at_percentile:.2f}')

    plt.show()
"""

"""逐列清理：
def iqr_od(df, num_cols, lambd):
    clean_df = df.copy()
    
    for col in num_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = lambd * clean_df[col].min() + (1 - lambd) * (Q1 - 1.5 * IQR)
        upper_bound = lambd * clean_df[col].max() + (1 - lambd) * (Q3 + 1.5 * IQR)
        
        # Filter out outliers
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
    
    return clean_df"""

"""grouped+清理
def iqr_od_grouped(df, num_cols, cat_cols, lambd):
    # Define a function to clean data in each group
    def clean_group(group):
        for col in num_cols:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = lambd * group[col].min() + (1 - lambd) * (Q1 - 1.5 * IQR)
            upper_bound = lambd * group[col].max() + (1 - lambd) * (Q3 + 1.5 * IQR)

            group = group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
        return group
    
    # Apply the cleaning function to each group
    return df.groupby(cat_cols).apply(clean_group).reset_index(drop=True)
"""

"""grouped+清理+标记outlier
def iqr_od_grouped(df, num_cols, cat_cols, lambd):
    # Define a function to clean data in each group
    def clean_group(group):
        for col in num_cols:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = lambd * group[col].min() + (1 - lambd) * (Q1 - 1.5 * IQR)
            upper_bound = lambd * group[col].max() + (1 - lambd) * (Q3 + 1.5 * IQR)

            # Update the outlier column for each numeric column
            if 'outlier' in group.columns:
                group['outlier'] |= ((group[col] < lower_bound) | (group[col] > upper_bound))
            else:
                group['outlier'] = ((group[col] < lower_bound) | (group[col] > upper_bound))

        return group

    # Apply the cleaning function to each group
    return df.groupby(cat_cols).apply(clean_group).reset_index(drop=True)
"""

