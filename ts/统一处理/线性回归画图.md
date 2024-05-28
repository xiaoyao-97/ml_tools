import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.graph_objects as go

def lr_plot(df, target, features, date_col):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df[[date_col, target] + features].copy()
    
    # Drop rows with missing values
    df_copy.dropna(inplace=True)
    
    # Extract X and y
    X = df_copy[features]
    y = df_copy[target]
    
    # Fit linear regression without intercept
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    
    # Calculate R-squared
    r2 = lr.score(X, y)
    
    # Print summary
    X_sm = sm.add_constant(X)  # Add constant term for statsmodels
    model = sm.OLS(y, X_sm)
    results = model.fit()
    print(results.summary())
    
    # Predictions
    df_copy['Prediction'] = lr.predict(X)
    
    # Plot real vs prediction
    fig_real_pred = go.Figure()
    fig_real_pred.add_trace(go.Scatter(x=df_copy[date_col], y=df_copy[target], mode='lines', name='Real'))
    fig_real_pred.add_trace(go.Scatter(x=df_copy[date_col], y=df_copy['Prediction'], mode='lines', name='Prediction'))
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
    df_copy['Residuals'] = df_copy[target] - df_copy['Prediction']
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_copy[date_col], y=df_copy['Residuals'], mode='lines', name='Residuals'))
    fig_res.update_layout(
        title='Residuals',
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
    fig_res.show()
    print(r2)
    return df_copy