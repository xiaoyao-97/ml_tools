# simple impute
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_constant = SimpleImputer(strategy='constant', fill_value=0)

df['A'] = imputer_mean.fit_transform(df[['A']])
df['B'] = imputer_median.fit_transform(df[['B']])
df['C'] = imputer_most_frequent.fit_transform(df[['C']])
df['A'] = imputer_constant.fit_transform(df[['A']])



# RF
from sklearn.ensemble import RandomForestRegressor

def rf_imp(df, cols):
    df_copy = df.copy()
    for col in cols:
        missing_mask = df_copy[col].isnull()
        
        if missing_mask.sum() == 0:
            continue  # 如果没有缺失值，跳过该列
        
        # 创建特征矩阵和目标变量
        X_train = df_copy.loc[~missing_mask, df_copy.columns != col]
        y_train = df_copy.loc[~missing_mask, col]
        X_test = df_copy.loc[missing_mask, df_copy.columns != col]
        
        # 删除所有列中含有缺失值的行
        combined = pd.concat([X_train, y_train], axis=1).dropna(axis=0, how='any')
        X_train = combined.drop(columns=[col])
        y_train = combined[col]
        
        # 构建和训练随机森林回归器
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 预测缺失值
        predicted_values = rf.predict(X_test)
        
        # 填补缺失值
        df_copy.loc[missing_mask, col] = predicted_values
        
    return df_copy

# RF cols and feature_cols
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def rf_imp(df, cols, feature_cols):
    df_copy = df.copy()  # 复制原数据框
    for col in cols:
        missing_mask = df_copy[col].isnull()  # 找到缺失值掩码
        
        if missing_mask.sum() == 0:
            continue  # 如果没有缺失值，跳过该列
        
        # 创建特征矩阵和目标变量
        current_feature_cols = [c for c in feature_cols if c != col]
        X_train = df_copy.loc[~missing_mask, current_feature_cols]
        y_train = df_copy.loc[~missing_mask, col]
        X_test = df_copy.loc[missing_mask, current_feature_cols]
        
        # 删除所有列中含有缺失值的行，但保留训练特征列
        combined = pd.concat([X_train, y_train], axis=1).dropna(subset=current_feature_cols)
        X_train = combined[current_feature_cols]
        y_train = combined[col]
        
        # 构建和训练随机森林回归器
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # 确保预测时特征列的顺序与训练时一致
        X_test = X_test[X_train.columns]
        
        # 预测缺失值
        predicted_values = rf.predict(X_test)
        
        # 检查预测值的长度是否与缺失值的数量一致
        if len(predicted_values) != missing_mask.sum():
            raise ValueError(f"Predicted values length ({len(predicted_values)}) does not match missing values count ({missing_mask.sum()}).")
        
        # 将预测值转换为Series，并对齐到原数据框索引
        predicted_series = pd.Series(predicted_values, index=df_copy.loc[missing_mask].index)
        
        # 填补缺失值
        df_copy.loc[missing_mask, col] = predicted_series
        
    return df_copy
"""

# simpler RF
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_missing), columns=df.columns)
original_data_scaled = scaler.transform(df)
df_rf_imputed = df_scaled.copy()
for column in df_scaled.columns:
    # 使用其他特征来预测缺失值
    missing_index = df_scaled[column].isna()
    if missing_index.sum() > 0:
        rf_imputer = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_imputer.fit(df_scaled.loc[~missing_index].drop(column, axis=1), df_scaled.loc[~missing_index, column])
        df_rf_imputed.loc[missing_index, column] = rf_imputer.predict(df_scaled.loc[missing_index].drop(column, axis=1))


# KNN imput
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def knn_imput(df, cols_for_dis, cols_to_imp, k):
    df_copy = df.copy()
    
    # 创建一个包含用于计算距离和需要填补的所有列的数据框
    impute_data = df_copy[cols_for_dis].copy()
    
    # 标准化数据
    scaler = StandardScaler()
    impute_data_scaled = scaler.fit_transform(impute_data)
    
    # 使用KNNImputer进行填补
    imputer = KNNImputer(n_neighbors=k)
    imputed_data_scaled = imputer.fit_transform(impute_data_scaled)
    
    # 将填补后的数据逆标准化
    imputed_data = scaler.inverse_transform(imputed_data_scaled)
    
    # 将填补后的数据转回数据框
    imputed_df = pd.DataFrame(imputed_data, columns=impute_data.columns)
    
    # 将填补后的列更新回原始数据框
    df_copy[cols_to_imp] = imputed_df[cols_to_imp]
    
    return df_copy



# lgb
df_lgb_imputed = df_scaled.copy()
for column in df_scaled.columns:
    missing_index = df_scaled[column].isna()
    if missing_index.sum() > 0:
        lgb_imputer = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        lgb_imputer.fit(df_scaled.loc[~missing_index].drop(column, axis=1), df_scaled.loc[~missing_index, column])
        df_lgb_imputed.loc[missing_index, column] = lgb_imputer.predict(df_scaled.loc[missing_index].drop(column, axis=1))

# xgb
df_xgb_imputed = df_scaled.copy()
for column in df_scaled.columns:
    missing_index = df_scaled[column].isna()
    if missing_index.sum() > 0:
        xgb_imputer = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_imputer.fit(df_scaled.loc[~missing_index].drop(column, axis=1), df_scaled.loc[~missing_index, column])
        df_xgb_imputed.loc[missing_index, column] = xgb_imputer.predict(df_scaled.loc[missing_index].drop(column, axis=1))




