scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_features(df, cols, n):
    features = df[cols]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=n)
    pca_components = pca.fit_transform(scaled_features)
    
    for i in range(n):
        df[f'pca_{i+1}'] = pca_components[:, i]
    
    return df
