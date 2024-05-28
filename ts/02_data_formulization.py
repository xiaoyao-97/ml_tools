""" 原始df变成matrix
import pandas as pd
df = train.copy()
date_col = 'date'
df['date'] = pd.to_datetime(df[date_col])
df['store_item'] = 'store_' + df['store'].astype(str) + '_item_' + df['item'].astype(str)
new_df = df.pivot_table(index='date', columns='store_item', values='sales', aggfunc='sum')
"""

""" 原始df变成matrix
def ts_to_matrix(df, date_col, matrix_cols, value_col, agg_func='sum'):
    df[date_col] = pd.to_datetime(df[date_col])
    df['combined'] = df[matrix_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
    new_df = df.pivot_table(index=date_col, columns='combined', values=value_col, aggfunc=agg_func)
    return new_df"""

"""matrix还原成df：
def matrix_to_df(new_df, date_col, value_col, matrix_cols):
    new_df = new_df.reset_index()
    melted_df = pd.melt(new_df, id_vars=[date_col], var_name='combined', value_name=value_col)
    melted_df[matrix_cols] = melted_df['combined'].str.split('_', expand=True)
    for col in matrix_cols:
        melted_df[col] = melted_df[col].astype(int)
    melted_df = melted_df.drop("combined", axis = 1)
    return melted_df
"""





