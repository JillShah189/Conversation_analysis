import streamlit as st
import json
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import plotly.graph_objs as go
from io import BytesIO

# Load the dataset
with open('c:/Users/admin/Downloads/CoMTA_dataset.json') as f:
    dataset = json.load(f)

def show_page2():
    def plot_combined_conversation_graph(dataset):
        G = nx.DiGraph()
        edge_weights = {}

        for conversation in dataset:
            previous_category = None

            for message in conversation['data']:
                current_category = message['category']
                node_size = 3000 + 100 * len(current_category)  # Adjust node size based on text length
                G.add_node(current_category, size=node_size)
                if previous_category:
                    if G.has_edge(previous_category, current_category):
                        G[previous_category][current_category]['weight'] += 1
                    else:
                        G.add_edge(previous_category, current_category, weight=1)
                previous_category = current_category

        # Collect edge weights
        edge_weights = nx.get_edge_attributes(G, 'weight')

        pos = nx.spring_layout(G, k=5, iterations=50)  # Adjust layout parameters
        node_sizes = [data['size'] for _, data in G.nodes(data=True)]

        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()}, node_size=node_sizes, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
        plt.title('Combined Adjacency Graph for All Conversations')

        # Return G and edge weights
        return G, edge_weights

# Function to create an interactive bar chart of edge weights
    def plot_edge_weights_bar_chart(edge_weights):
        edge_labels = [' -> '.join(edge) for edge in edge_weights.keys()]
        edge_values = list(edge_weights.values())

        trace = go.Bar(
            x=edge_labels,
            y=edge_values,
            marker=dict(color='blue')
        )

        layout = go.Layout(
            title='Edge Weights Bar Chart',
            xaxis=dict(title='Edges'),
            yaxis=dict(title='Weight'),
            margin=dict(l=40, r=40, t=40, b=140),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace], layout=layout)
        return fig

    # Streamlit app
    st.title("Combined Conversation Graph and Edge Weights Analysis")

    st.subheader("Combined Graph Visualization")
    G, edge_weights = plot_combined_conversation_graph(dataset)
    st.pyplot(plt.gcf())  # Display the matplotlib figure

    st.subheader("Interactive Edge Weights Bar Chart")
    fig = plot_edge_weights_bar_chart(edge_weights)
    st.plotly_chart(fig)  # Display the Plotly interactive bar chart
