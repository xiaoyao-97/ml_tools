"""线性填充
from sklearn.linear_model import LinearRegression

def linear_fill(df, col, condition_fill, condition_train, n, linear_factor):
    df_sorted = df.sort_values(by=linear_factor).reset_index(drop=True)
    
    for idx, row in df_sorted[condition_fill].iterrows():
        before_idx = df_sorted[condition_train & (df_sorted.index < idx)].tail(n).index
        after_idx = df_sorted[condition_train & (df_sorted.index > idx)].head(n).index
        valid_indices = before_idx.union(after_idx)
        
        if len(valid_indices) < n:
            continue
        
        X = df_sorted.loc[valid_indices, linear_factor].values.reshape(-1, 1)
        y = df_sorted.loc[valid_indices, col].values
        
        if len(X) < 2:
            continue
        
        model = LinearRegression()
        model.fit(X, y)
        
        df_sorted.at[idx, col] = model.predict([[row[linear_factor]]])[0]
    
    return df_sorted
"""






