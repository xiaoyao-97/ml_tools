"""预测图：
def ts_plot(df1, df2, time_col1, time_col2, col1, col2, pred, point_size=1, line_width=0.1):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot df1 col1 with customized point size and line width
    ax.plot(np.array(df1[time_col1]), np.array(df1[col1]), label=f'Real {col1}', 
            color='blue', marker='o', linestyle='-', markersize=point_size, linewidth=line_width)
    
    # Plot df2 col2 and pred with customized point size and line width
    ax.plot(np.array(df2[time_col2]), np.array(df2[col2]), label=f'Real {col2}', 
            color='green', marker='x', linestyle='--', markersize=point_size, linewidth=line_width)
    ax.plot(np.array(df2[time_col2]), np.array(df2[pred]), label=f'Predicted {pred}', 
            color='red', marker='s', linestyle='-.', markersize=point_size, linewidth=line_width)
    
    # Set legend
    ax.legend()
    
    # Set title and axis labels
    ax.set_title('Time Series Plot')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    
    # Display grid
    ax.grid(True)
    
    # Display the plot
    plt.show()
"""

"""交互式预测图：
import plotly.graph_objects as go

def ts_plot(df1, df2, time_col1, time_col2, col1, col2, pred, point_size=1, line_width=0.1):
    # Create a new figure
    fig = go.Figure()
    fig.update_layout(
        width=15500,
        height=800
    )


    # Plot df1 col1 with customized point size and line width
    fig.add_trace(go.Scatter(x=df1[time_col1], y=df1[col1], mode='lines+markers',
                             name=f'Real {col1}', line=dict(color='blue', width=line_width),
                             marker=dict(size=point_size), 
                             hoverinfo='x+y', 
                             hovertemplate=f"Time: %{{x|%Y-%m-%d %H:%M:%S}}<br>{col1}: %{{y}}"))

    # Plot df2 col2 and pred with customized point size and line width
    fig.add_trace(go.Scatter(x=df2[time_col2], y=df2[col2], mode='lines+markers',
                             name=f'Real {col2}', line=dict(color='green', dash='dash', width=line_width),
                             marker=dict(size=point_size, symbol='x'), 
                             hoverinfo='x+y', 
                             hovertemplate=f"Time: %{{x|%Y-%m-%d %H:%M:%S}}<br>{col2}: %{{y}}"))

    fig.add_trace(go.Scatter(x=df2[time_col2], y=df2[pred], mode='lines+markers',
                             name=f'Predicted {pred}', line=dict(color='red', dash='dot', width=line_width),
                             marker=dict(size=point_size, symbol='square'), 
                             hoverinfo='x+y', 
                             hovertemplate=f"Time: %{{x|%Y-%m-%d %H:%M:%S}}<br>{pred}: %{{y}}"))

    # Set legend, title, and axis labels
    fig.update_layout(title='Time Series Plot', xaxis_title='Date and Time', yaxis_title='Values',
                      legend_title="Legend", hovermode='closest', plot_bgcolor='white')

    # Add grid lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='LightPink')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='LightBlue')

    # Display the plot
    fig.show()
"""

"""residual图：
def plot_residual(df, col, pred, pointsize=20):
    # Calculate residuals
    df['residual'] = df[col] - df[pred]
    
    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(df[col], df['residual'], s=pointsize)
    plt.title('Residual Plot')
    plt.xlabel(col)
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()
"""

"""悬停交互式residual：
import plotly.express as px
def plot_residual(df, time_col, col, pred, pointsize=5):
    # Calculate residuals
    df['residual'] = df[pred] - df[col]
    
    # Plotting the residuals using Plotly
    fig = px.scatter(df, x=col, y='residual',
                     width=800, height=480,
                     hover_data={col: True, 'residual': True, time_col: True},
                     title='Residual Plot')

    # Update marker size
    fig.update_traces(marker=dict(size=pointsize))

    fig.update_layout(xaxis_title=col,
                      yaxis_title='Residuals',
                      plot_bgcolor='white')

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightBlue')

    fig.show()
"""

"""滑动窗口：
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

def plot_ts(df, time_col, cols, point_size=5, line_size=2, fig_width=800, fig_height=600):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='time-series-plot'),
        dcc.Checklist(
            id='series-checklist',
            options=[{'label': col, 'value': col} for col in cols],
            value=cols,
            inline=True
        )
    ])

    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('series-checklist', 'value')]
    )
    def update_figure(selected_series):
        fig = go.Figure([
            go.Scatter(
                x=df[time_col], y=df[col], mode='lines+markers', name=col,
                marker=dict(size=point_size), line=dict(width=line_size)
            ) for col in selected_series
        ])

        fig.update_layout(
            xaxis_title='Time', yaxis_title='Value', hovermode='x unified',
            width=fig_width, height=fig_height,
            title_text="Time series with range slider and selectors",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True), type="date"
            )
        )

        return fig

    app.run_server(debug=True)
"""



