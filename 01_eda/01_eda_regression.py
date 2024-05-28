"""target-数值变量
def plot_averaged_y(df, x_col, y_col, d):
    # 确定 x 的最大最小值
    min_x = df[x_col].min()
    max_x = df[x_col].max()

    # 创建一个间隔为 d/3 的网格
    grid = np.arange(min_x, max_x, d/3)

    # 使用列表来存储临时数据，最后一起合并
    rows = []

    # 对每个网格点进行循环
    for x in grid:
        # 找到在 (x-d, x+d) 范围内的 y 值
        y_vals = df[(df[x_col] >= x - d) & (df[x_col] <= x + d)][y_col]
        # 计算平均值
        mean_y = y_vals.mean()
        # 添加到列表
        rows.append({x_col: x, 'Average ' + y_col: mean_y})

    # 使用 concat 合并所有行
    result = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(result[x_col], result['Average ' + y_col], marker='o')
    plt.title(f'Average of {y_col} within ±{d} around grid points with interval {d/3}')
    plt.xlabel(x_col)
    plt.ylabel('Average ' + y_col)
    plt.grid(True)
    plt.show()
"""


