from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# 创建随机数据集
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 初始化AIC和BIC列表
aic_scores = []
bic_scores = []

# 尝试不同的成分数量，并计算对应的AIC和BIC分数
for n in range(1, 11):
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))
    bic_scores.append(gmm.bic(X))

# 绘制AIC和BIC分数曲线
plt.plot(range(1, 11), aic_scores, label='AIC')
plt.plot(range(1, 11), bic_scores, label='BIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.legend()
plt.show()

# 选择最优的成分数量
best_n_components_aic = np.argmin(aic_scores) + 1
best_n_components_bic = np.argmin(bic_scores) + 1

print(f'Best number of components (AIC): {best_n_components_aic}')
print(f'Best number of components (BIC): {best_n_components_bic}')

# 使用最优的成分数量进行聚类
best_gmm_aic = GaussianMixture(n_components=best_n_components_aic)
best_gmm_aic.fit(X)
labels_aic = best_gmm_aic.predict(X)

best_gmm_bic = GaussianMixture(n_components=best_n_components_bic)
best_gmm_bic.fit(X)
labels_bic = best_gmm_bic.predict(X)

# 绘制聚类结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_aic, s=40, cmap='viridis')
plt.title(f'GMM Clustering with {best_n_components_aic} Components (AIC)')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_bic, s=40, cmap='viridis')
plt.title(f'GMM Clustering with {best_n_components_bic} Components (BIC)')

plt.show()