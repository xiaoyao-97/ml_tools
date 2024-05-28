# 很好的演示网站：https://pair-code.github.io/understanding-umap/

# ——————————————————————tsne——————————————————————————————————
from sklearn.manifold import TSNE

def tsne_features(df, cols, n, random_state=42):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])
    tsne = TSNE(n_components=n, random_state=random_state)
    tsne_result = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(tsne_result, columns=[f'tsne_{i+1}' for i in range(n)])
    return pd.concat([df, tsne_df], axis=1)



# ———————————————————————UMAP———————————————————————————————————
import umap
from sklearn.preprocessing import StandardScaler

def umap_features(df, cols, n, random_state=42):
    # Scale the specified columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols])
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_components=n, random_state=random_state)
    umap_result = umap_reducer.fit_transform(scaled_data)
    
    # Create a DataFrame with the UMAP results
    umap_df = pd.DataFrame(umap_result, columns=[f'umap_{i+1}' for i in range(n)])
    
    # Concatenate the original DataFrame with the UMAP results
    return pd.concat([df, umap_df], axis=1)


# ——————————————————————LDA————————————————————————————————
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def lda_features(df, cols, target_col, n):
    lda = LDA(n_components=n)
    lda_result = lda.fit_transform(df[cols], df[target_col])
    lda_df = pd.DataFrame(lda_result, columns=[f'lda_{i+1}' for i in range(n)])
    return pd.concat([df, lda_df], axis=1)