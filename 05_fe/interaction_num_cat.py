"""一些哲学
如果cat0与num0完全独立，则任何interaction变量都是没有意义的。


cat0与num0的交互有以下几种：
1. 不同的cat0，相对num0有不同的线性模型；
2. 不同的cat0，num0的分布有所不同
    1. num0的均值有所不同


"""

# first level interaction： cat0不同的num0分布不同
def calculate_r2_with_cat_mean(df, cat_col, num_col):
    from sklearn.metrics import r2_score
    """
    Calculate the R^2 of predicting a numeric variable from the means of a categorical variable.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    cat_col (str): Name of the categorical column.
    num_col (str): Name of the numeric column.
    
    Returns:
    float: The R^2 score of the predictions.
    """
    category_means = df.groupby(cat_col)[num_col].mean()
    
    df['predicted'] = df[cat_col].map(category_means)
    
    r2 = r2_score(df[num_col], df['predicted'])
    return r2

def calculate_r2_with_cat_mean_cv(df, cat_col, num_col, n_splits=5):
    """
    Calculate the R^2 of predicting a numeric variable from the means of a categorical variable using cross-validation.
    
    Args:
    df (pd.DataFrame): DataFrame containing the data.
    cat_col (str): Name of the categorical column.
    num_col (str): Name of the numeric column.
    n_splits (int): Number of folds for KFold cross-validation.
    
    Returns:
    float: The average R^2 score of the predictions across the cross-validation folds.
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []

    for train_index, test_index in kf.split(df):
        # Splitting the data into train and test sets
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        
        # Creating a copy of test_df to avoid SettingWithCopyWarning
        test_df = test_df.copy()
        
        # Calculating means on the training set
        category_means = train_df.groupby(cat_col)[num_col].mean()
        
        # Using .loc to ensure we're modifying a copy
        test_df.loc[:, 'predicted'] = test_df[cat_col].map(category_means)
        
        # Filling missing predictions with the global mean of the training set
        global_mean = train_df[num_col].mean()
        test_df.loc[:, 'predicted'].fillna(global_mean, inplace=True)
        
        # Calculating R^2 score for the current fold
        r2 = r2_score(test_df[num_col], test_df['predicted'])
        r2_scores.append(r2)
    
    # Returning the average R^2 score across all folds
    return sum(r2_scores) / len(r2_scores)

