import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.sankey import Sankey

def plot_isotope_network(G, highlight_nodes=None, title="Isotope Transmutation Network"):
    pos = nx.spring_layout(G)
    node_colors = ['red' if n in (highlight_nodes or []) else 'skyblue' for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=800)
    plt.title(title)
    plt.show()

def plot_time_evolution(times, inventories, isotope_names):
    plt.figure(figsize=(10,6))
    for idx, name in enumerate(isotope_names):
        plt.plot(times, inventories[:, idx], label=name)
    plt.xlabel('Time (days)')
    plt.ylabel('Inventory (atoms or mass)')
    plt.legend()
    plt.title('Isotope Inventory Evolution')
    plt.show()

def plot_optimization_progress(history):
    steps, costs = zip(*history)
    plt.plot(steps, costs, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Optimization Progress')
    plt.show()

def plot_sankey_flows(flows, labels, title="Isotope Flow Sankey Diagram"):
    """
    Plot a Sankey diagram for isotope flows.
    flows: list of flow values (positive for input, negative for output)
    labels: list of labels for each flow
    """
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    sankey = Sankey(ax=ax, unit=None)
    sankey.add(flows=flows, labels=labels, orientations=[0]*len(flows))
    plt.title(title)
    plt.show()

# Interactive network plot stub
import plotly.graph_objects as go

def plot_interactive_network(G, title="Interactive Isotope Network"):
    """
    Plot an interactive isotope network using Plotly.
    """
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), marker=dict(size=10, color='skyblue'))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=title, showlegend=False))
    fig.show()

def get_3d_network_figure(G, title="3D Isotope Network"):
    """
    Return a Plotly 3D network figure for use in Streamlit.
    """
    import plotly.graph_objects as go
    pos = nx.spring_layout(G, dim=3)
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='gray', width=2))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_z = [pos[n][2] for n in G.nodes()]
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text', text=list(G.nodes()), marker=dict(size=8, color='skyblue'))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False)
    return fig

def plot_3d_network(G, title="3D Isotope Network"):
    """
    Plot a 3D interactive isotope network using Plotly (opens in browser or Streamlit if available).
    """
    fig = get_3d_network_figure(G, title)
    try:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        fig.show()

# Time-lapse animation stub
def animate_time_evolution(times, inventories, isotope_names):
    """
    Stub for time-lapse animation of isotope inventories.
    """
    print("[Animation] Time-lapse animation not yet implemented.") 