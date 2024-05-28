"""robust scaler
def robust(df):
    from sklearn.preprocessing import RobustScaler
    robust_scaler = RobustScaler()
    scaled_data = robust_scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df 
    
def robust(df,columns_to_scale):
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_columns = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), columns=columns_to_scale, index=df.index)
    scaled_df = pd.concat([scaled_columns, df.drop(columns_to_scale, axis=1)], axis=1)
    return scaled_df
    """

"""standard scaler
def standard(df):
    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler()
    scaled_data = standard_scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df
"""

"""minmax scaler
def minmax(df):
    from sklearn.preprocessing import MinMaxScaler
    minmax_scaler = MinMaxScaler()
    scaled_data = minmax_scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df
"""


"""spline
def spline(df):
    from sklearn.preprocessing import SplineTransformer
    spline_transformer = SplineTransformer(degree=3, n_knots=3)
    scaled_data = spline_transformer.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df
"""
