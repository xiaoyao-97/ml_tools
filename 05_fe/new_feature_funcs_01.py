# https://www.kaggle.com/code/lucasmorin/time-series-agregation-functions

""" 所有的函数
def median(x):
    ''' 计算中位数 '''
    return np.median(x)

def mean(x):
    ''' 计算均值 '''
    return np.mean(x)

def length(x):
    ''' 返回数组长度 '''
    return len(x)

def standard_deviation(x):
    ''' 计算标准差 '''
    return np.std(x)

def large_standard_deviation(x):
    ''' 计算归一化标准差，当最大值与最小值相等时返回NaN '''
    if (np.max(x)-np.min(x)) == 0:
        return np.nan
    else:
        return np.std(x)/(np.max(x)-np.min(x))

def variation_coefficient(x):
    ''' 计算变异系数（标准差除以均值） '''
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan

def variance_std_ratio(x):
    ''' 计算方差与其标准差之比 '''
    y = np.var(x)
    if y != 0:
        return y/np.sqrt(y)
    else:
        return np.nan

def ratio_beyond_r_sigma(x, r):
    ''' 计算超过r倍标准差的比例 '''
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size

def range_ratio(x):
    ''' 计算范围比例，即均值与中位数之差除以最大值与最小值之差 '''
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference
    
def has_duplicate_max(x):
    ''' 检查数组中是否有重复的最大值 '''
    return np.sum(x == np.max(x)) >= 2

def has_duplicate_min(x):
    ''' 检查数组中是否有重复的最小值 '''
    return np.sum(x == np.min(x)) >= 2

def has_duplicate(x):
    ''' 检查数组中是否有重复值 '''
    return x.size != np.unique(x).size

def count_duplicate_max(x):
    ''' 计算数组中最大值的重复次数 '''
    return np.sum(x == np.max(x))

def count_duplicate_min(x):
    ''' 计算数组中最小值的重复次数 '''
    return np.sum(x == np.min(x))

def count_duplicate(x):
    ''' 计算数组中重复值的数量 '''
    return x.size - np.unique(x).size

def sum_values(x):
    ''' 计算数组中所有值的总和 '''
    if len(x) == 0:
        return 0
    return np.sum(x)

def log_return(list_stock_prices):
    ''' 计算股票价格的对数回报率 '''
    return np.log(list_stock_prices).diff() 

def realized_volatility(series):
    ''' 计算实现波动率 '''
    return np.sqrt(np.sum(series**2))

def realized_abs_skew(series):
    ''' 计算绝对偏度的立方根 '''
    return np.power(np.abs(np.sum(series**3)),1/3)

def realized_skew(series):
    ''' 计算偏度的立方根，保留符号 '''
    return np.sign(np.sum(series**3))*np.power(np.abs(np.sum(series**3)),1/3)

def realized_vol_skew(series):
    ''' 计算六次方根的波动率偏度 '''
    return np.power(np.abs(np.sum(series**6)),1/6)

def realized_quarticity(series):
    ''' 计算四次方根的峰度 '''
    return np.power(np.sum(series**4),1/4)

def count_unique(series):
    ''' 计算数组中唯一值的数量 '''
    return len(np.unique(series))

def count(series):
    ''' 计算数组中的元素数量 '''
    return series.size

# 最大回撤函数
def maximum_drawdown(series):
    ''' 计算最大回撤 '''
    series = np.asarray(series)
    if len(series)<2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]

count_above_0 = lambda x: count_above(x,0)
count_above_0.__name__ = 'count_above_0'

count_below_0 = lambda x: count_below(x,0)
count_below_0.__name__ = 'count_below_0'

value_count_0 = lambda x: value_count(x,0)
value_count_0.__name__ = 'value_count_0'

count_near_0 = lambda x: range_count(x,-0.00001,0.00001)
count_near_0.__name__ = 'count_near_0_0'

ratio_beyond_01_sigma = lambda x: ratio_beyond_r_sigma(x,0.1)
ratio_beyond_01_sigma.__name__ = 'ratio_beyond_01_sigma'

ratio_beyond_02_sigma = lambda x: ratio_beyond_r_sigma(x,0.2)
ratio_beyond_02_sigma.__name__ = 'ratio_beyond_02_sigma'

ratio_beyond_03_sigma = lambda x: ratio_beyond_r_sigma(x,0.3)
ratio_beyond_03_sigma.__name__ = 'ratio_beyond_03_sigma'

number_crossing_0 = lambda x: number_crossing_m(x,0)
number_crossing_0.__name__ = 'number_crossing_0'

quantile_01 = lambda x: quantile(x,0.1)
quantile_01.__name__ = 'quantile_01'

quantile_025 = lambda x: quantile(x,0.25)
quantile_025.__name__ = 'quantile_025'

quantile_075 = lambda x: quantile(x,0.75)
quantile_075.__name__ = 'quantile_075'

quantile_09 = lambda x: quantile(x,0.9)
quantile_09.__name__ = 'quantile_09'

number_peaks_2 = lambda x: number_peaks(x,2)
number_peaks_2.__name__ = 'number_peaks_2'

mean_n_absolute_max_2 = lambda x: mean_n_absolute_max(x,2)
mean_n_absolute_max_2.__name__ = 'mean_n_absolute_max_2'

number_peaks_5 = lambda x: number_peaks(x,5)
number_peaks_5.__name__ = 'number_peaks_5'

mean_n_absolute_max_5 = lambda x: mean_n_absolute_max(x,5)
mean_n_absolute_max_5.__name__ = 'mean_n_absolute_max_5'

number_peaks_10 = lambda x: number_peaks(x,10)
number_peaks_10.__name__ = 'number_peaks_10'

mean_n_absolute_max_10 = lambda x: mean_n_absolute_max(x,10)
mean_n_absolute_max_10.__name__ = 'mean_n_absolute_max_10'

get_first = lambda x: x.iloc[0]
get_first.__name__ = 'get_first'

get_last = lambda x: x.iloc[-1]
get_last.__name__ = 'get_last'
"""

""" 函数的名称
base_stats = [mean,sum,length,standard_deviation,variation_coefficient,variance,skewness,kurtosis]
higher_order_stats = [abs_energy,root_mean_square,sum_values,realized_volatility,realized_abs_skew,realized_skew,realized_vol_skew,realized_quarticity]
min_median_max = [minimum,median,maximum]
additional_quantiles = [quantile_01,quantile_025,quantile_075,quantile_09]
other_min_max = [absolute_maximum,max_over_min,max_over_min_sq]
min_max_positions = [last_location_of_maximum,first_location_of_maximum,last_location_of_minimum,first_location_of_minimum]
peaks = [number_peaks_2, mean_n_absolute_max_2, number_peaks_5, mean_n_absolute_max_5, number_peaks_10, mean_n_absolute_max_10]
counts = [count_unique,count,count_above_0,count_below_0,value_count_0,count_near_0]
reoccuring_values = [count_above_mean,count_below_mean,percentage_of_reoccurring_values_to_all_values,percentage_of_reoccurring_datapoints_to_all_datapoints,sum_of_reoccurring_values,sum_of_reoccurring_data_points,ratio_value_number_to_time_series_length]
count_duplicate = [count_duplicate,count_duplicate_min,count_duplicate_max]
variations = [mean_abs_change,mean_change,mean_second_derivative_central,absolute_sum_of_changes,number_crossing_0]
ranges = [variance_std_ratio,ratio_beyond_01_sigma,ratio_beyond_02_sigma,ratio_beyond_03_sigma,large_standard_deviation,range_ratio]
get_first = [get_first, get_last]

all_functions = base_stats + higher_order_stats + min_median_max + additional_quantiles + other_min_max + min_max_positions + peaks + counts + variations + ranges 
"""

""" transform
from scipy.stats import skew, kurtosis
funcs = [np.mean, np.std, np.median, skew, kurtosis]

def agg_func(df, cat_cols, num_cols, funcs, func_names):
    original_df = df.copy()
    
    for cat_col in cat_cols:
        for num_col in num_cols:
            for i,func in enumerate(funcs):
                df[cat_col+'_'+num_col+'_'+func_names[i]] = df.groupby(cat_col)[num_col].transform(func)
    return df
"""

""" 更复杂的agg函数
def xminusmean(s):
    return s - np.mean(s)


funcs = [np.mean, np.std, np.median, xminusmean]
func_names = ['mean', 'std', 'median', 'x-m']

def agg_func2(df, cat_cols, num_cols, funcs, func_names):
    df = df.copy()  # 创建副本，避免修改原始数据

    for cat_col in cat_cols:
        for num_col in num_cols:
            for i, func in enumerate(funcs):
                func_name = func_names[i]
                new_col_name = f'{cat_col}_{num_col}_{func_name}'
                df[new_col_name] = df.groupby(cat_col)[num_col].transform(func)
    
    return df
"""


"""target funct
def target_func(df, cat_col, num_col, row_name, func, lambda_val=2):
    # Step 1: Group by cat_col and apply func on num_col for 'train' rows
    train_df = df[df['type'] == 'train']
    groupby_result = train_df.groupby(cat_col)[num_col].apply(func).reset_index()
    groupby_result.columns = [cat_col, row_name]
    
    # Step 2: Merge the result back to the original dataframe
    df = df.merge(groupby_result, on=cat_col, how='left')
    
    # Step 3: Handle rows where type != 'train' and cat_col value is not in train
    overall_counts = df[cat_col].value_counts()
    train_counts = train_df[cat_col].value_counts()
    
    # Calculate the threshold based on lambda
    threshold = lambda_val * overall_counts.loc[train_counts.index]
    
    # Filter rows in train that appear less than the threshold
    rare_values = train_counts[train_counts < threshold]
    
    # Calculate the mean of num_col for these rare values
    rare_values_mean = func(train_df[train_df[cat_col].isin(rare_values.index)][num_col])
    
    # Assign the mean value to rows where type != 'train' and cat_col is missing in train
    df.loc[df[cat_col].isin(overall_counts.index.difference(train_counts.index)) & (df['type'] != 'train'), row_name] = rare_values_mean
    
    return df
"""


