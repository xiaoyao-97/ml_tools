results = []
for trial in trials:
    result = {
        'params': trial['misc']['vals'],
        'loss': trial['result']['loss'],
        # 可以根据需要添加其他结果
    }
    results.append(result)

df_results = pd.DataFrame(results)

pd.set_option('display.max_colwidth', None)
df_results.sort_values("loss")