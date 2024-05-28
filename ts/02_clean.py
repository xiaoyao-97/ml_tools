"""判断是否所有的组合都有值

total_combinations = df['Date'].nunique() * df['store'].nunique() * df['product'].nunique()

actual_combinations = df.groupby(['Date', 'store', 'product']).size().reset_index(name='Count').shape[0]

if total_combinations == actual_combinations:
    print("所有可能的组合都至少有一条记录。")
else:
    print(f"存在缺失的组合。总共应有 {total_combinations} 组合，但实际只有 {actual_combinations} 组合。")
    
————————————————————————————————如果没有，把缺失的行找出来————————————————

dates = df['Date'].unique()
stores = df['store'].unique()
products = df['product'].unique()

all_combinations = pd.MultiIndex.from_product([dates, stores, products], names=['Date', 'store', 'product'])
all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()

existing_combinations = df[['Date', 'store', 'product']].drop_duplicates()

missing_combinations = pd.merge(all_combinations_df, existing_combinations, on=['Date', 'store', 'product'], how='left', indicator=True)
missing_combinations = missing_combinations[missing_combinations['_merge'] == 'left_only']
missing_combinations = missing_combinations.drop(columns=['_merge'])

print(missing_combinations) 
"""




