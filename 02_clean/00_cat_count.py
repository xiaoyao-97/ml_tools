from itertools import product

def count_combinations(df, cat_cols):
    unique_values = [df[col].unique() for col in cat_cols]

    all_combinations = list(product(*unique_values))

    combinations_df = pd.DataFrame(all_combinations, columns=cat_cols)

    combinations_df['count'] = combinations_df.apply(lambda row: len(df[(df[cat_cols] == row[cat_cols].values).all(axis=1)]), axis=1)

    return combinations_df