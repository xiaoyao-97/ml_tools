"""画出missing value"""
def missing_value_visualization(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import missingno as msno
    colors = ['black','white']
    # colors = ['white','black']
    sns.heatmap(df.isnull(), cmap=sns.color_palette(colors))
    msno.matrix(df)

"""msno.dendrogram(df) 把missing value做hierarchical clustering"""


"""打印missing value数量及比例"""
def missing_value_count(df):
    missing_val_count_by_column = (df.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])


# 打印nan： rows_with_nan = df[df.isna().any(axis=1)]
# rows_with_special_values = df[(df.isna().any(axis=1)) | (df == np.inf).any(axis=1) | (df == -np.inf).any(axis=1)]


