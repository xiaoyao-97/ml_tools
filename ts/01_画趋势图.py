def plot_item_price_trend(df, item_id):
    # 筛选出特定item_id的数据
    item_data = df[df['item_id'] == item_id]
    
    # 将date列转换为日期类型
    item_data['date'] = pd.to_datetime(item_data['date'])
    
    # 按日期排序
    item_data = item_data.sort_values('date')
    
    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(item_data['date'], item_data['item_price'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Item Price')
    plt.title(f'Price Trend for Item ID {item_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_col(df, col):
    import matplotlib.pyplot as plt
    time_col = 'date_time'
    df[time_col] = pd.to_datetime(df[time_col])
    
    df = df.sort_values(time_col)
    
    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    plt.scatter(df[time_col], df[col], marker='o')
    plt.xlabel(time_col)
    plt.ylabel(col)
    # plt.title(f'Price Trend for Item ID {item_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 对每个group画走势图
def plot_sales_trends(data, col_date, group_features, value_col):
    data[col_date] = pd.to_datetime(data[col_date])
    
    grouped = data.groupby(group_features)
    
    # 为每个组绘制走势图
    for (store, product), group in grouped:
        plt.figure(figsize=(10, 5))
        plt.plot(group[col_date], group[value_col])
        # plt.title(f'Sales Trend for Store {store}, Product {product}')
        plt.xlabel(col_date)
        plt.ylabel(value_col)
        plt.grid(True)
        plt.show()

# 更具有交互性的作图：
import plotly.graph_objects as go
import pandas as pd
train_wh = train_wh[~train_wh['date'].apply(lambda x: (x.month, x.day) in holidays)]
for keys, group in train_wh.groupby(matrix_cols):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=group['date'], y=group['log2'], mode='lines', name=f'{keys}'))
    fig.update_layout(
        title=f'Log2 Values Over Time for {keys}',
        xaxis_title='Date',
        yaxis_title='Log2 Value',
        showlegend=True
    )
    fig.show()


# 在同一张图上画出预测和实际的趋势：
df['ds'] = pd.to_datetime(df['ds'])  # 将日期列转换为 datetime 类型
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Actual y')
plt.plot(df['ds'], df['yhat'], label='Predicted yhat')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

"""【重要】ploty交互式
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

def plot_ts(df, time_col, cols, point_size=5, line_size=2, fig_width=800, fig_height=600):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='time-series-plot'),
        dcc.RangeSlider(
            id='time-window-slider',
            min=0,
            max=len(df)-1,
            value=[0, len(df)-1],
            marks={i: str(df[time_col].iloc[i]) for i in range(0, len(df), max(1, len(df)//10))}
        ),
        dcc.Checklist(
            id='series-checklist',
            options=[{'label': col, 'value': col} for col in cols],
            value=cols,
            inline=True
        )
    ])

    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('time-window-slider', 'value'),
         Input('series-checklist', 'value')]
    )
    def update_figure(selected_time_window, selected_series):
        filtered_df = df.iloc[selected_time_window[0]:selected_time_window[1]+1]

        fig = go.Figure()

        for col in selected_series:
            fig.add_trace(go.Scatter(
                x=filtered_df[time_col],
                y=filtered_df[col],
                mode='lines+markers',
                name=col,
                marker=dict(size=point_size),
                line=dict(width=line_size)
            ))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            width=fig_width,
            height=fig_height
        )

        return fig

    app.run_server(debug=True)
"""

"""显示display_cols的ploty：
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_ts(df, time_col, cols, display_cols, point_size=5, line_size=2, fig_width=800, fig_height=600):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='time-series-plot'),
        dcc.RangeSlider(
            id='time-window-slider',
            min=0,
            max=len(df)-1,
            value=[0, len(df)-1],
            marks={i: str(df[time_col].iloc[i]) for i in range(0, len(df), max(1, len(df)//10))}
        ),
        dcc.Checklist(
            id='series-checklist',
            options=[{'label': col, 'value': col} for col in cols],
            value=cols,
            inline=True
        )
    ])

    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('time-window-slider', 'value'),
         Input('series-checklist', 'value')]
    )
    def update_figure(selected_time_window, selected_series):
        filtered_df = df.iloc[selected_time_window[0]:selected_time_window[1]+1]

        fig = go.Figure()

        for col in selected_series:
            hover_text = filtered_df.apply(
                lambda row: f'{col}: {row[col]:.2f}<br>' +
                            '<br>'.join([f'{d_col}: {row[d_col]:.2f}' for d_col in display_cols]), 
                axis=1
            )
            fig.add_trace(go.Scatter(
                x=filtered_df[time_col],
                y=filtered_df[col],
                mode='lines+markers',
                name=col,
                marker=dict(size=point_size),
                line=dict(width=line_size),
                hoverinfo='text',
                hovertext=hover_text
            ))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            width=fig_width,
            height=fig_height
        )

        return fig

    app.run_server(debug=True)
"""

"""min-max画图并有rolling window
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

def plot_ts(df, time_col, cols, point_size=5, line_size=2, fig_width=800, fig_height=600):
    # 对所有列进行Min-Max归一化
    df_normalized = df.copy()
    for col in cols:
        df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

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
        fig = go.Figure()

        for i, col in enumerate(selected_series):
            fig.add_trace(go.Scatter(
                x=df[time_col], y=df_normalized[col], mode='lines+markers', name=col,
                text=[f"{col}: {val:.2f}" for val in df[col]],  # 在悬停提示中显示原始值
                hovertemplate='%{x}<br>%{text}<extra></extra>',
                marker=dict(size=point_size), line=dict(width=line_size)
            ))

        fig.update_layout(
            xaxis_title='Time', yaxis_title='Normalized Value', hovermode='x unified',
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
            ),
            yaxis=dict(range=[0, 1])  # 统一设置y轴范围为0到1
        )

        return fig

    app.run_server(debug=True)

# 示例调用
# df = pd.read_csv('your_data.csv')
# plot_ts(df, 'date_column', ['col1', 'col2', 'col3'])
"""

"""最基本的交互画图
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
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
        fig = go.Figure()

        for i, col in enumerate(selected_series):
            fig.add_trace(go.Scatter(
                x=df[time_col], y=df[col], mode='lines+markers', name=col,
                text=[f"{col}: {val:.2f}" for val in df[col]],  # 在悬停提示中显示原始值
                hovertemplate='%{x}<br>%{text}<extra></extra>',
                marker=dict(size=point_size), line=dict(width=line_size)
            ))

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

# 示例数据
data = {
    'time': pd.date_range(start='1/1/2020', periods=100, freq='D'),
    'series1': range(100),
    'series2': range(100, 200),
    'series3': range(200, 300)
}
df = pd.DataFrame(data)

plot_ts(df, 'time', ['series1', 'series2', 'series3'])
"""

