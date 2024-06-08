# rolling window
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def exp_decay_weights(length, halflife):
    alpha = np.log(2) / halflife
    return np.exp(-alpha * np.arange(length)[::-1])

def weighted_rolling_window_lr(df, target, features, window_length):
    # Initialize the output DataFrame
    results = pd.DataFrame(index=df.index, columns=features + ['intercept', 'predicted_' + target])
    
    # Prepare the features and target arrays
    X = df[features].values
    y = df[target].values
    
    # Initialize an array to store the predicted values
    predictions = np.full(len(df), np.nan)
    
    # Iterate through the dataset to compute rolling regression with exponential weights
    for end in range(window_length, len(df)):
        window_data = df.iloc[:end]
        
        # Calculate exponential decay weights
        weights = exp_decay_weights(len(window_data), window_length)
        weights /= weights.sum()
        
        # Fit the linear regression model using the weights
        model = LinearRegression()
        model.fit(window_data[features], window_data[target], sample_weight=weights)
        
        # Get the beta values and intercept
        betas = model.coef_
        intercept = model.intercept_
        
        # Predict the target for the current row
        current_features = df.iloc[end][features].values
        predicted_value = np.dot(current_features, betas) + intercept
        predictions[end] = predicted_value
        
        # Store the results
        results.iloc[end, :len(features)] = betas
        results.iloc[end, -2] = intercept
        results.iloc[end, -1] = predicted_value
    
    # Calculate metrics for the predictions
    valid_indices = ~np.isnan(predictions)
    rmse = mean_squared_error(y[valid_indices], predictions[valid_indices], squared=False)
    variance = np.var(predictions[valid_indices])
    r2 = r2_score(y[valid_indices], predictions[valid_indices])
    print('r2',r2)
    return results


# 没有常数项的版本（不预测过去）
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

def exp_decay_weights(length, window_length):
    alpha = 2 / (window_length + 1)
    weights = np.exp(-alpha * np.arange(length)[::-1])
    return weights

def weighted_rolling_window_lr_no_intercept(df, target, features, window_length, date_col, start_point=0, step=1):
    df_cleaned = df.dropna(subset=[target] + features).copy()
    
    results = pd.DataFrame(index=df_cleaned.index, columns=features + ['predicted_' + target])
    
    X = df_cleaned[features].values
    y = df_cleaned[target].values
    
    predictions = np.full(len(df_cleaned), np.nan)
    
    model = LinearRegression(fit_intercept=False)
    betas = np.zeros(len(features))
    
    # Iterate through the dataset to compute rolling regression with exponential weights
    for end in range(start_point + window_length, len(df_cleaned), step):
        window_data = df_cleaned.iloc[end - window_length:end]
        
        weights = exp_decay_weights(len(window_data), window_length)
        weights /= weights.sum()
        
        model.fit(window_data[features], window_data[target], sample_weight=weights)
        
        betas = model.coef_
        
        for idx in range(end, min(end + step, len(df_cleaned))):
            current_features = df_cleaned.iloc[idx][features].values
            predicted_value = np.dot(current_features, betas)
            predictions[idx] = predicted_value
        
        results.iloc[end-1, :len(features)] = betas
        if end < len(df_cleaned):
            results.iloc[end, -1] = predictions[end]
    
    # Assign the predictions back to the DataFrame
    df_cleaned['Prediction'] = predictions
    df_cleaned['Residual'] = df_cleaned[target] - df_cleaned['Prediction']
    
    # Calculate metrics for the predictions
    valid_indices = ~np.isnan(predictions)
    rmse = mean_squared_error(y[valid_indices], predictions[valid_indices], squared=False)
    variance = np.var(predictions[valid_indices])
    r2 = r2_score(y[valid_indices], predictions[valid_indices])
    print('r2', r2)
    
    # Plot real vs prediction
    fig_real_pred = go.Figure()
    fig_real_pred.add_trace(go.Scatter(x=df_cleaned[date_col], y=df_cleaned[target], mode='lines', name='Real'))
    fig_real_pred.add_trace(go.Scatter(x=df_cleaned[date_col], y=df_cleaned['Prediction'], mode='lines', name='Prediction'))
    fig_real_pred.update_layout(
        title='Real vs Prediction',
        xaxis_title=date_col,
        yaxis_title=target,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig_real_pred.show()
    
    # Plot residuals
    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(x=df_cleaned[date_col], y=df_cleaned['Residual'], mode='lines', name='Residual'))
    fig_residuals.update_layout(
        title='Residuals over Time',
        xaxis_title=date_col,
        yaxis_title='Residuals',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig_residuals.show()
    
    return results



"""最后的rolling window
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def exp_decay_weights(length, halflife):
    alpha = np.log(2) / halflife
    return np.exp(-alpha * np.arange(length)[::-1])

def weighted_rolling_window_lr(df, target, features, halflife, window, min_window):
    # Initialize the output DataFrame
    results = pd.DataFrame(index=df.index, columns=features + ['intercept', 'predicted_' + target], dtype=float)
    
    # Prepare the features and target arrays
    X = df[features].values
    y = df[target].values
    
    # Initialize an array to store the predicted values
    predictions = np.full(len(df), np.nan)
    
    # Iterate through the dataset to compute rolling regression with exponential weights
    for end in range(min_window, len(df)):
        start = max(0, end - window)
        window_data = df.iloc[start:end]
        
        # Calculate exponential decay weights
        weights = exp_decay_weights(len(window_data), halflife)
        weights /= weights.sum()
        
        # Fit the linear regression model using the weights
        model = LinearRegression()
        model.fit(window_data[features], window_data[target], sample_weight=weights)
        
        # Get the beta values and intercept
        betas = model.coef_
        intercept = model.intercept_
        
        # Predict the target for the current row
        current_features = df.iloc[end][features].values
        predicted_value = np.dot(current_features, betas) + intercept
        predictions[end] = predicted_value
        
        # Store the results
        results.iloc[end, :len(features)] = betas
        results.iloc[end, -2] = intercept
        results.iloc[end, -1] = predicted_value
    
    # Calculate metrics for the predictions
    valid_indices = ~np.isnan(predictions)
    rmse = mean_squared_error(y[valid_indices], predictions[valid_indices], squared=False)
    variance = np.var(predictions[valid_indices])
    r2 = r2_score(y[valid_indices], predictions[valid_indices])
    print('R²:', r2)
    print('RMSE:', rmse)
    print('Variance:', variance)
    
    return results"""
