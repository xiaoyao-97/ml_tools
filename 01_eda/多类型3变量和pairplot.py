"""三个col
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import Delaunay


def plot_data(df, col1, col2, col3):
    # Determine the types of columns
    col1_type = df[col1].dtype
    col2_type = df[col2].dtype
    col3_type = df[col3].dtype
    
    if np.issubdtype(col1_type, np.number) and np.issubdtype(col2_type, np.number) and np.issubdtype(col3_type, np.number):
        # All three columns are numeric, plot contour plot
        plot_contour(df, col1, col2, col3)
    elif (np.issubdtype(col1_type, np.number) and not np.issubdtype(col2_type, np.number)) or (not np.issubdtype(col1_type, np.number) and np.issubdtype(col2_type, np.number)):
        # One column is numeric and the other is categorical, plot scatter plot
        plot_scatter(df, col1, col2, col3)
    elif not np.issubdtype(col1_type, np.number) and not np.issubdtype(col2_type, np.number):
        # Both columns are categorical, plot box plot with hue
        plot_box(df, col1, col2, col3)
    else:
        raise ValueError("Unsupported column types")

def plot_scatter(df, col1, col2, col3):
    if np.issubdtype(df[col1].dtype, np.number):
        numeric_col = col1
        category_col = col2
    else:
        numeric_col = col2
        category_col = col1
    
    plt.figure()
    sns.scatterplot(data=df, x=numeric_col, y=col3, hue=category_col)
    plt.xlabel(numeric_col)
    plt.ylabel(col3)
    plt.title(f'Scatter plot of {numeric_col} vs {col3} colored by {category_col}')
    plt.show()
    
            
def plot_contour(df, col1, col2, col3):
    xi = np.linspace(df[col1].min(), df[col1].max(), 100)
    yi = np.linspace(df[col2].min(), df[col2].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((df[col1], df[col2]), df[col3], (xi, yi), method='linear')

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, zi, levels=15, cmap='viridis')
    plt.colorbar(contour, label=f'{col3} Mean')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Contour Plot of {col3} Mean')
    plt.show()

def plot_box(df, col1, col2, col3):
    plt.figure()
    sns.boxplot(data=df, x=col1, y=col3, hue=col2)
    plt.xlabel(col1)
    plt.ylabel(col3)
    plt.title(f'Box plot of {col3} based on {col1} and {col2}')
    plt.show()
"""

"""用上面的画pairplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata

def plot_data(df, col1, col2, col3, ax):
    # Determine the types of columns
    col1_type = df[col1].dtype
    col2_type = df[col2].dtype
    col3_type = df[col3].dtype
    
    if np.issubdtype(col1_type, np.number) and np.issubdtype(col2_type, np.number) and np.issubdtype(col3_type, np.number):
        # All three columns are numeric, plot contour plot
        plot_contour(df, col1, col2, col3, ax)
    elif (np.issubdtype(col1_type, np.number) and not np.issubdtype(col2_type, np.number)) or (not np.issubdtype(col1_type, np.number) and np.issubdtype(col2_type, np.number)):
        # One column is numeric and the other is categorical, plot scatter plot
        plot_scatter(df, col1, col2, col3, ax)
    elif not np.issubdtype(col1_type, np.number) and not np.issubdtype(col2_type, np.number):
        # Both columns are categorical, plot box plot with hue
        plot_box(df, col1, col2, col3, ax)
    else:
        raise ValueError("Unsupported column types")

def plot_contour(df, col1, col2, col3, ax):
    # 生成网格数据
    xi = np.linspace(df[col1].min(), df[col1].max(), 100)
    yi = np.linspace(df[col2].min(), df[col2].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # 对 col3 数据进行插值
    zi = griddata((df[col1], df[col2]), df[col3], (xi, yi), method='linear')

    # 绘制等势线图
    contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
    plt.colorbar(contour, ax=ax, label=f'{col3} Mean')
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f'Contour Plot of {col3} Mean')

# def plot_scatter(df, col1, col2, col3, ax):
#     sns.scatterplot(data=df, x=col1, y=col2, hue=col3, ax=ax)
#     ax.set_xlabel(col1)
#     ax.set_ylabel(col2)
#     ax.set_title(f'Scatter plot of {col1} vs {col2} colored by {col3}')
    
def plot_scatter(df, col1, col2, col3, ax):
    if np.issubdtype(df[col1].dtype, np.number):
        numeric_col = col1
        category_col = col2
    else:
        numeric_col = col2
        category_col = col1
    
    sns.scatterplot(data=df, x=numeric_col, y=col3, hue=category_col, ax=ax)
    ax.set_xlabel(numeric_col)
    ax.set_ylabel(col3)
    ax.set_title(f'Scatter plot of {numeric_col} vs {col3} colored by {category_col}')

def plot_box(df, col1, col2, col3, ax):
    sns.boxplot(data=df, x=col1, y=col3, hue=col2, ax=ax)
    ax.set_xlabel(col1)
    ax.set_ylabel(col3)
    ax.set_title(f'Box plot of {col3} based on {col1} and {col2}')

# 需要绘制的列
cols = ['MSZoning', 'LandContour', 'GrLivArea', 'GarageArea']

# 创建一个图形网格
fig, axes = plt.subplots(len(cols), len(cols), figsize=(15, 15))

# 遍历每一对列，并在对应的子图中绘制图形
for i, col1 in enumerate(cols):
    for j, col2 in enumerate(cols):
        ax = axes[i, j]  # 获取当前的子图
        if col1 != col2:
            plot_data(df, col1, col2, 'SalePrice', ax)
        else:
            # 如果是对角线上的图（即 col1 == col2），则只显示列名
            ax.text(0.5, 0.5, col1, ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

# 调整子图之间的间距
plt.tight_layout()
plt.show()
"""






