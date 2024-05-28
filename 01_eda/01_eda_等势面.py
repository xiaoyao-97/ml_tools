"""三角等势面
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def plot_contours(df, x_col, y_col, z_col):
    # 提取数据点
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    # 创建三角网格和绘图
    triang = Triangulation(x, y)
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(triang, z, levels=15, cmap='viridis')
    plt.colorbar(contour, label='z_col values')
    plt.scatter(x, y, color='skyblue', marker='o', label='Data points',s = 05)  # 显示原始数据点
    plt.legend()
    plt.title(f'Contour Plot of {z_col} Using Triangulation')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
"""

"""svr等势面
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def plot_svr_contour(df, x_col, y_col, z_col, sample=1, C=1.0,  epsilon=0.1, resolution=500):
    '''
    使用支持向量机回归(SVR)在XY平面上对Z值绘制等势面图。

    参数:
        df (DataFrame): 包含数据的DataFrame。
        x_col (str): 作为X轴特征的列名。
        y_col (str): 作为Y轴特征的列名。
        z_col (str): 目标变量列名。
        C (float): SVR模型的C参数，默认为1.0。
        epsilon (float): SVR模型的epsilon参数，默认为0.1。
        resolution (int): 网格的分辨率，默认500（网格将是500x500）。
    '''
    df = df.sample(n=int(sample*len(df)),random_state = 1)
    # 提取特征和目标变量
    X = df[[x_col, y_col]].values
    y = df[z_col].values.ravel()

    # 创建SVR模型，包括数据预处理
    model = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon))

    # 训练模型
    model.fit(X, y)

    # 创建预测网格
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    print(x_min, x_max, y_min, y_max)
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), 
                                 np.linspace(y_min, y_max, resolution))
    grid_X = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    # 预测网格点的z值
    z_pred = model.predict(grid_X).reshape(x_grid.shape)

    # 绘制等势面图
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x_grid, y_grid, z_pred, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Contour Plot of Predicted Z values on XY plane')
    plt.show()

# 示例使用方式
# 假设df是已经加载的DataFrame，包含列'x_col', 'y_col', 'z_col'
# plot_svr_contour(df, 'x_col', 'y_col', 'z_col')
"""

"""poly等势面
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def plot_poly_contour(df, x_col, y_col, z_col, degree=2, resolution=500):
    '''
    使用多项式回归在XY平面上对Z值绘制等势面图。

    参数:
        df (DataFrame): 包含数据的DataFrame。
        x_col (str): 作为X轴特征的列名。
        y_col (str): 作为Y轴特征的列名。
        z_col (str): 目标变量列名。
        degree (int): 多项式的度数，默认为2。
        resolution (int): 网格的分辨率，默认500（网格将是500x500）。
    '''
    # 提取特征和目标变量
    X = df[[x_col, y_col]].values
    y = df[z_col].values.ravel()

    # 创建多项式回归模型，包括数据预处理
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

    # 训练模型
    model.fit(X, y)

    # 创建预测网格
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), 
                                 np.linspace(y_min, y_max, resolution))
    grid_X = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    # 预测网格点的z值
    z_pred = model.predict(grid_X).reshape(x_grid.shape)

    # 绘制彩色等势面图
    plt.figure(figsize=(8, 6))
    contourf = plt.contourf(x_grid, y_grid, z_pred, levels=50, cmap='viridis')
    plt.colorbar(contourf)

    # 在彩色等势面图上绘制黑色实线等势线
    contour = plt.contour(x_grid, y_grid, z_pred, levels=50, colors='black')
    plt.scatter(df[x_col], df[y_col], color='skyblue', marker='o', label='Data points',s = 0.5)  # 显示原始数据点
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Contour Plot of Predicted Z values on XY plane')
    plt.show()

# 示例使用方式
# 假设df是已经加载的DataFrame，包含列'x_col', 'y_col', 'z_col'
# plot_poly_contour(df, 'x_col', '
"""


