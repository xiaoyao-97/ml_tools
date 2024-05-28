def pca(df, cols, n_components,s = ""):
    from sklearn.decomposition import PCA
    X = df[cols]
    
    pca_model = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(X)
    
    pca_df = pd.DataFrame(data=pca_result, columns=['pca_'+s+str(i+1) for i in range(n_components)])
    df[pca_df.columns] = pca_df
    
    return df