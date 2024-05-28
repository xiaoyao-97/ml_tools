检验
1. 基于分布：
   1. 3sigma
   2. boxplot
   3. grubbs假设检验
   4. Z-score
2. 基于分类
   1. one-class svm
3. 基于聚类
   1. DBSCAN
4. 基于距离
   1. KNN
5. 基于密度
   1. COF
   2. LOF
   3. SOS
6. 基于降维
   1. PCA
   2. autoencoder
7. 基于预测
   1. ARIMA
   2. Moving average
8. 基于树：
   1. iforest

第一：模型的使用
第二：模型的不同参数
第三：多个模型的组合
第四：模型的验证（pipeline）

pyod的模型：
   iforest
   KNN(average, mean)
   LOF(CBLOF)
   MCD (QMCD)
   OCSVM (One-class SVM)
   HBOS
   ABOD
   (FB)
   INNE
   Feature Bagging
   SUOD
   PCA (KPCA)

标准化：
KNN (K-Nearest Neighbors) - KNN算法在计算点之间的距离时依赖于各个特征的尺度，因此标准化对于性能和结果的准确性至关重要。
LOF (Local Outlier Factor) - LOF算法也依赖于距离的计算，因此推荐对数据进行标准化。
ABOD (Angle-Based Outlier Detection) - ABOD通过考虑数据点之间的角度分布来检测异常值，标准化可以帮助避免一个维度对结果的过度影响。
OCSVM (One-Class SVM) - 这种方法依赖于数据的核转换，通常需要标准化以避免某些特征在核转换时占优势。
PCA (Principal Component Analysis) / KPCA (Kernel PCA) - PCA和KPCA都涉及到数据的方差结构，标准化可以确保所有特征对主成分分析的贡献是均衡的。
MCD (Minimum Covariance Determinant) - 在计算协方差时，标准化有助于使特征尺度一致，从而更准确地估计“中心”和“分散”。
以下模型对数据标准化的需求不那么强烈：

HBOS (Histogram-based Outlier Score) - 通常不需要标准化，因为它基于单维度的直方图来评估异常。
Feature Bagging - 这取决于所使用的基础模型；如果基础模型是距离或方差敏感的（如KNN或LOF），则可能需要标准化。
SUOD (Scalable Unsupervised Outlier Detection) - 该框架本身不需要特定的数据处理，但其下属的模型可能需要，具体取决于选择的算法。
INNE - 对标准化的需求视具体实现而定，但通常在处理距离相关的计算时标准化会有所帮助。
   