
# 时间特征
def add_time_features(df, col="datetime"):
    df['hour'] = df[col].dt.hour
    df['dayofweek'] = df[col].dt.dayofweek
    df['quarter'] = df[col].dt.quarter
    df['month'] = df[col].dt.month
    df['year'] = df[col].dt.year
    df['dayofyear'] = df[col].dt.dayofyear
    df['dayofmonth'] = df[col].dt.day
    df['weekofyear'] = df[col].dt.weekofyear
    return df

# 时间统计量
# 判断格式是否可以转换：
date_format = '%d.%m.%Y'
def check_date_format(date_str):
    from datetime import datetime
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False
train['is_valid'] = train['date'].apply(check_date_format)

from datetime import datetime
date_datetime = datetime.strptime(date_string, "%d.%m.%Y")
train['date'] = pd.to_datetime(train['date']) # 这一行是必须的
# 或者直接：
sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')

# 转变为月份
train['date'] = pd.to_datetime(train['date'])
train['month'] = train['date'].dt.to_period('M')


# 对一个特征
#求平均：
avg_prices = df.groupby('item_id')['item_price'].mean()
# 把值记录在原来的df上
average_prices = train.groupby('item_id')['item_price'].transform('mean')
train['price_ratio'] = train['item_price']/average_prices

"""一些特征：
上个月的销售总额
一个月某商品的销售总额
一个月某店的销售总额
商品类别
店+商品类别
店+子类别
城市
商品+城市
"""
# 每月求和特征
train2['monthly_item_cnt'] = train2.groupby(['month', 'item_id'])['item_cnt_day'].transform('sum')

# shift
df['lagged_item_cnt'] = df.groupby(['item_id', 'shop_id'])['item_cnt_day'].shift(1)

# 节日特征
def add_is_day_col(data, month, day):
    data[f'{month}-{day}'] = data['date'].apply(lambda x: 1 if (x.month == month) and (x.day == day) else 0)
    return data
