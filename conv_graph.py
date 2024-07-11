import json
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import plotly.graph_objs as go
from io import BytesIO  # Import BytesIO

# Load the dataset
with open('c:/Users/admin/Downloads/CoMTA_dataset.json') as f:
    dataset = json.load(f)
def show_page1():
    # Extract all test IDs
    test_ids = [conversation['test_id'] for conversation in dataset]

    # Streamlit app
    st.title("Conversation Graph Explorer")

    # Dropdown to select conversation by test_id
    selected_test_id = st.selectbox("Select Test ID", test_ids)

    # Multiselect for filtering categories
    all_categories = set()
    for conversation in dataset:
        for message in conversation['data']:
            all_categories.add(message['category'])
    selected_categories = st.multiselect("Select Categories to Display", sorted(all_categories), default=sorted(all_categories))

    # Function to plot using igraph
    def plot_igraph(conversation):
        nodes = []
        edges = []
        edge_weights = {}

        previous_category = None
        
        for message in conversation['data']:
            current_category = message['category']
            if current_category not in nodes:
                nodes.append(current_category)
            if previous_category:
                edge = (previous_category, current_category)
                if edge in edge_weights:
                    edge_weights[edge] += 1
                else:
                    edges.append(edge)
                    edge_weights[edge] = 1
            previous_category = current_category

        G = ig.Graph(directed=True)
        G.add_vertices(nodes)
        G.add_edges(edges)
        G.es['weight'] = [edge_weights[edge] for edge in edges]

        layout = G.layout("fr")
        fig, ax = plt.subplots(figsize=(10, 8))
        visual_style = {
            "vertex_label": nodes,
            "edge_width": [G.es['weight'][i] for i in range(len(edges))],
            "vertex_color": 'skyblue',
            "vertex_size": 50,
            "vertex_label_size": 10,
            "edge_arrow_size": 0.5,
            "layout": layout,
            "bbox": (800, 800),
            "margin": 40,
            "target": ax
        }
        ig.plot(G, **visual_style)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf
    '''
    for edge in edges:
            x0, y0 = layout[nodes.index(edge[0])]
            x1, y1 = layout[nodes.index(edge[1])]
            weight = edge_weights[edge]
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(weight), fontsize=10, ha='center', va='center', color='red')
    '''
        # Add edge weight annotations
        
        

    # Function to plot using networkx and Plotly for interactivity
    def plot_networkx_plotly(conversation):
        G = nx.DiGraph()
        previous_category = None
        
        for message in conversation['data']:
            current_category = message['category']
            if current_category in selected_categories:
                G.add_node(current_category)
                if previous_category and previous_category in selected_categories:
                    if G.has_edge(previous_category, current_category):
                        G[previous_category][current_category]['weight'] += 1
                    else:
                        G.add_edge(previous_category, current_category, weight=1)
                previous_category = current_category

        pos = nx.spring_layout(G)
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=weight, color='blue'),
                hoverinfo='none',
                mode='lines'
            )
            edge_trace.append(trace)
            
            # Add text annotation for edge weight
            edge_trace.append(go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                text=[str(weight)],
                mode='text',
                textposition='middle center',
                hoverinfo='none'
            ))

        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=list(G.nodes()),
            mode='markers+text',
            textposition='bottom center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                color='skyblue',
                size=30,
                line=dict(width=2)
            )
        )

        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            annotations=[dict(
                text="Networkx Graph with Plotly",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.1,
                xanchor="center", yanchor="top"
            )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
        return fig

    # Get the conversation for the selected test_id
    conversation = next(conv for conv in dataset if conv['test_id'] == selected_test_id)

    # Plot graphs
    st.subheader("igraph Visualization")
    st.image(plot_igraph(conversation))

    st.subheader("networkx Visualization with Plotly Interactivity")
    fig = plot_networkx_plotly(conversation)
    st.plotly_chart(fig)
