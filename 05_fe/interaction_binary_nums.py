def interaction_pairs_logit(df, y, num_cols):
    import statsmodels.api as sm
    from itertools import combinations
    import pandas as pd
    import numpy as np
    results = []

    # Iterate over all combinations of two columns
    for (col1, col2) in combinations(num_cols, 2):
        temp_df = df.copy()
        # Create an interaction term between the two columns
        temp_df['Interaction'] = temp_df[col1] * temp_df[col2]
        # Add a constant for logistic regression
        X_with_const = sm.add_constant(temp_df[[col1, col2, 'Interaction']])
        # Fit a logistic regression model
        model = sm.Logit(y, X_with_const)
        results_model = model.fit(disp=0)  # disp=0 suppresses the fit summary output
        # Extract the p-value of the interaction term
        p_value_interaction = results_model.pvalues['Interaction']

        results.append({
            'Variable Pair': f'{col1} & {col2}',
            'feature1': col1,
            'feature2': col2,
            'P-Value of Interaction': p_value_interaction if pd.notnull(p_value_interaction) else np.nan
        })

    # Convert results to a DataFrame and process
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=['P-Value of Interaction'])  # Drop rows where p-value is NaN
    results_df = results_df.sort_values(by='P-Value of Interaction')

    return results_df