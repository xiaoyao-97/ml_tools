import plotly.express as px

def pair_od(df,cols):
    fig = px.scatter_matrix(df,
        dimensions=cols,
        color='outlier',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_data=['Index']) 
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(width=1200, height=800)
    fig.update_traces(diagonal_visible=False)
    fig.show()




