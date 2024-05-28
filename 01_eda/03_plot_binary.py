# target的数量分布
fig = px.histogram(df, x='target')
fig.update_layout(
    title_text='Target distribution', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
)
fig.show()




# 数值变量的kdeplot
n_rows, n_cols = 4,3
f, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))
f.suptitle('Distribution of Features', fontsize=16)

for index, column in enumerate(df[num_cols].columns):
    i,j = (index // n_cols, index % n_cols)
    sns.kdeplot(train.loc[train['target'] == 0, column], color="m", shade=True, ax=axes[i,j])
    sns.kdeplot(train.loc[train['target'] == 1, column], color="b", shade=True, ax=axes[i,j])

f.delaxes(axes[3, 2])
plt.tight_layout()
plt.show()



# 分类变量与target的interaction：
train_0_df = train.loc[train['target'] == 0]
train_1_df = train.loc[train['target'] == 1]
n_rows, n_cols = 10,2
fig = make_subplots(rows=n_rows, cols=n_cols)
for index, column in enumerate(df[cat_cols].columns):
    i,j = ((index // n_cols)+1, (index % n_cols)+1)
    data = train_0_df.groupby(column)[column].count().sort_values(ascending=False)
    data = data if len(data) < 10 else data[:10]
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 0',
    ), row=i, col=j)

    data = train_1_df.groupby(column)[column].count().sort_values(ascending=False)
    data = data if len(data) < 10 else data[:10]
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 1'
    ), row=i, col=j)
    
    fig.update_xaxes(title=column, row=i, col=j)
    fig.update_layout(barmode='stack')
    
fig.update_layout(
    autosize=False,
    width=1000,
    height=3600,
    showlegend=False,
)
fig.show()