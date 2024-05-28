"""支持向量回归（SVR）是一种非常强大的回归方法，适用于多种数据类型和应用场景，特别是在以下几种情况中表现较好：

非线性关系：SVR 特别擅长处理目标变量和特征之间存在复杂非线性关系的数据。通过选择合适的核函数（如径向基（RBF）核、多项式核等），SVR 能够在高维空间中有效地处理这些非线性关系。
高维数据：SVR 在处理高维特征空间（特征数量很多）时表现良好，因为它的性能不太受维度诅咒（dimensionality curse）的影响。这得益于它的核技巧，该技巧允许它在高维空间中进行计算，而无需显式地映射数据到这些高维空间。
小样本数据：对于样本量相对较小的数据集，SVR 也能够表现出良好的性能。它的优化目标侧重于找到一个平衡好的决策边界，以避免过拟合，这在样本量不足时尤为重要。
噪声数据：通过调整其正则化参数 C 和误差容忍度 epsilon，SVR 可以较好地处理噪声数据。C 控制了模型的复杂度和训练误差的权衡，而 epsilon 控制了模型对于预测误差的敏感性，可以忽略掉目标值的微小变化，这使得模型对噪声不太敏感。
然而，SVR 也有一些局限性和不适合的情况：

大规模数据集：SVR 的计算复杂度相对较高，尤其是在数据点数量很多时。对于大规模数据集，训练一个 SVR 模型可能会非常耗时。
稀疏数据：虽然 SVR 可以处理高维数据，但如果数据非常稀疏，SVR 的性能可能不如专门设计来处理稀疏数据的算法。"""

"""C：正则化参数。C的值越大，模型对每个数据点的拟合就越严格，可能导致过拟合。较小的C值可以增加模型的泛化能力，但可能导致欠拟合。

gamma：核函数的系数。对于RBF，这是选择RBF函数的宽度的参数，影响了数据映射到新特征空间的粒度。

epsilon：设置在预测中允许的不精确程度。较小的epsilon值意味着创建更精确的模型，可能导致过拟合。"""


"""gridsearch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
import numpy as np
import pandas as pd

class ProgressDisplay:
    def __init__(self, param_grid, n_splits):
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.total_runs = n_splits * np.prod([len(v) for v in param_grid.values()])
        self.current_run = 0

    def update(self, params, fold, score):
        self.current_run += 1
        progress = (self.current_run / self.total_runs) * 100
        print(f"Params: {params} | Fold: {fold + 1}/{self.n_splits} | Loss: {score:.4f} | Progress: {progress:.2f}%")

def custom_scorer(display, estimator, X, y):
    y_pred = estimator.predict(X)
    loss = np.sqrt(np.sum((y - y_pred) ** 2))
    fold = display.current_run % display.n_splits
    params = estimator.get_params()
    display.update(params, fold, loss)
    return -loss  

model = SVR(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1],
    'epsilon': [0.1, 0.5, 1]
}
cv = 5
random_state = 1
kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

progress_display = ProgressDisplay(param_grid, cv)

grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=lambda estimator, X, y: custom_scorer(progress_display, estimator, X, y))
grid_search.fit(train.drop(columns=[target]), train[target])

results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score', 'std_test_score']]
print(results)
"""

"""optuna+cv
import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-1, 100.0)  # 正则化参数
    epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1.0)  # Epsilon in the epsilon-SVR model
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1)  # Kernel coefficient for 'rbf'

    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    return np.mean(score)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # 可以调整n_trials来控制搜索的迭代次数

best_params = study.best_params
print('Best parameters:', best_params)

best_svr = SVR(kernel='rbf', **best_params)
best_svr.fit(X_train, y_train)

y_pred = best_svr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
"""








