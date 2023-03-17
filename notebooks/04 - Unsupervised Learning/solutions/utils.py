import plotly.graph_objs as go

def plot_3d(coords, labels, title):
    # Create trace
    trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels,
            colorscale='Viridis',
            opacity=0.8
        )
    )

    # Create layout
    layout = go.Layout(
        margin=dict(l=20, r=0, b=0, t=30),
        title=title,
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)
    
    # Show plot
    fig.show()


def plot_2d(X, y, title, ax):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), loc="best", title="Clusters")
    ax.add_artist(legend)