# Define the number of rows and columns
nrows = 3
ncols = 4

# Create a figure and a subplot grid
fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10), sharex=False)

# Iterate over columns and create boxplots
for i in range(nrows):
    for j in range(ncols):
        col = data[plt_cols].columns[i * ncols + j]
        sns.histplot(ax=axs[i, j], data=data[col])
        axs[i, j].set_title(col)

# Adjust layout and show the plot
fig.tight_layout()
plt.show()


