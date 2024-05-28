# pip install dataframe-to-image
from dataframe_to_image import dataframe_to_image
dataframe_to_image.convert(df,visualisation_library='matplotlib')


# pip install dataframe-image
import dataframe_image as dfi
dfi.export(df, 'dataframe_image.png')


# plt
def save_as_img(df,s):
    fig, ax = plt.subplots(figsize=(10, 2))  # 调整图表大小
    ax.axis('off')  # 不显示轴
    table = tbl.Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = df.shape
    width, height = 1.0 / ncols, 1.0 / (nrows + 1)
    for i in range(ncols):
        table.add_cell(0, i, width, height, text=df.columns[i], loc='center', facecolor='lightgrey')
    for i in range(nrows):
        for j in range(ncols):
            table.add_cell(i + 1, j, width, height, text=df.iloc[i, j], loc='center', facecolor='white')
    ax.add_table(table)
    plt.savefig(s+'.png', bbox_inches='tight')
    plt.show()
save_as_img(df,'traial01')



