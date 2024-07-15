import json
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import plotly.graph_objs as go
from io import BytesIO  # Import BytesIO
from pyvis.network import Network 

# Load the dataset
with open('CoMTA_dataset.json') as f:
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
        # Add edge weight annotations
    def plot_igraph(conversation):
        nodes = []
        edges = []
        node_freq = {}
        edge_weights = {}

        previous_category = None
        
        for message in conversation['data']:
            current_category = message['category']
            if current_category not in nodes:
                nodes.append(current_category)
                node_freq[current_category] = 1
            else:
                node_freq[current_category] += 1
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

        layout = G.layout("fr")  # Use Kamada-Kawai layout for curved edges
        fig, ax = plt.subplots(figsize=(14, 10))
        base_node_size = 75  # Set a base size for nodes
        visual_style = {
            "vertex_label": nodes,
            "edge_width": [G.es['weight'][i] for i in range(len(edges))],
            "vertex_color": 'skyblue',
            "vertex_size": [base_node_size + node_freq[node] * 10 for node in nodes],  # Base size plus frequency adjustment
            "vertex_label_size": 12,
            "edge_arrow_size": 5,  # Make arrows larger to ensure visibility
            "layout": layout,
            "bbox": (800, 800),
            "margin": 40,
            "target": ax
        }
        
        ig.plot(G, **visual_style)
        
        # Add edge labels on the lines with adjustments to avoid overlap
        for edge in G.es:
            source, target = edge.tuple
            weight = edge['weight']
            midpoint = [(layout[source][i] + layout[target][i]) / 2 for i in range(2)]
            offset = 1  # Adjust the offset for better spacing
            ax.text(midpoint[0], midpoint[1], str(weight), fontsize=10 + weight / 5, color='black', ha='center', va='center', backgroundcolor='white')

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf

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
    # Function to plot using pyvis for interactivity
    def plot_pyvis(conversation):
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

        net = Network(notebook=True, directed=True)
        for node in G.nodes():
            net.add_node(node, label=node, title=node, color='skyblue', size=30)

        for edge in G.edges(data=True):
            src, dst, data = edge
            weight = data['weight']
            net.add_edge(src, dst, value=weight, title=str(weight))

        net.show_buttons(filter_=['physics'])
        net.save_graph('network.html')
        return 'network.html'
    
    # Get the conversation for the selected test_id
    conversation = next(conv for conv in dataset if conv['test_id'] == selected_test_id)

    # Plot graphs
    st.subheader("igraph Visualization")
    st.image(plot_igraph(conversation))
    st.write("The start node is User Queries and the end node is User Answer")

    st.subheader("networkx Visualization with Plotly Interactivity")
    fig = plot_networkx_plotly(conversation)
    st.plotly_chart(fig)

    st.subheader("pyvis Interactive Visualization")
    pyvis_html = plot_pyvis(conversation)
    st.components.v1.html(open(pyvis_html, 'r').read(), height=600)
