import dask.dataframe as dd

ddf = dd.from_pandas(data[plt_cols], npartitions=4)
sample_ddf = ddf.sample(frac=0.01, random_state=42).compute()
sns.pairplot(sample_ddf, kind='kde')
