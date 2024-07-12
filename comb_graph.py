'''import json
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
from pyvis.network import Network 

# Load the dataset
with open('CoMTA_dataset.json') as f:
    dataset = json.load(f)

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

def plot_edge_weights_bar_chart(edge_weights):
    edge_labels = [' -> '.join(edge) for edge in edge_weights.keys()]
    edge_values = list(edge_weights.values())

    trace = go.Bar(
        y=edge_labels,  # Reverse x and y axis assignments
        x=edge_values,  # Reverse x and y axis assignments
        orientation='h',  # Specify horizontal orientation
        marker=dict(color='blue')
    )

    layout = go.Layout(
        title='Edge Weights Bar Chart',
        xaxis=dict(title='Weight'),  # Adjust x-axis label
        yaxis=dict(title='Edges'),  # Adjust y-axis label
        margin=dict(l=40, r=40, t=40, b=140),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig

def plot_pyvis_graph(dataset):
    G = nx.DiGraph()
    previous_category = None

    for conversation in dataset:
        for message in conversation['data']:
            current_category = message['category']
            if current_category not in G:
                G.add_node(current_category)
            if previous_category:
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
    net.save_graph('pyvis_graph.html')  # Save the HTML locally
    return 'pyvis_graph.html'  # Return the filename

# Streamlit app
def show_page2():
    st.title("Combined Conversation Graph and Edge Weights Analysis")

    # Extract all unique math levels
    all_math_levels = sorted(set([conversation['math_level'] for conversation in dataset]))

    # Dropdown to select math levels
    selected_math_levels = st.multiselect("Select Math Levels to Display", all_math_levels, default=all_math_levels)

    if selected_math_levels:
        # Filter dataset based on selected math levels
        filtered_dataset = [conversation for conversation in dataset if conversation['math_level'] in selected_math_levels]

        if filtered_dataset:
            st.subheader("NetworkX Combined Graph Visualization")
            G, edge_weights = plot_combined_conversation_graph(filtered_dataset)
            st.pyplot(plt.gcf())  # Display the matplotlib figure

            st.subheader("Pyvis Interactive Network Graph")
            pyvis_html = plot_pyvis_graph(filtered_dataset)
            st.components.v1.html(open(pyvis_html, 'r').read(), height=800)

            st.subheader("Interactive Edge Weights Bar Chart")
            fig = plot_edge_weights_bar_chart(edge_weights)
            st.plotly_chart(fig)  # Display the Plotly interactive bar chart
        else:
            st.write("No conversations found for the selected math levels.")
    else:
        st.write("Please select at least one math level.")

# Display the Streamlit app
show_page2()
'''

import streamlit as st
import json
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go

# Load the dataset
with open('CoMTA_dataset.json') as f:
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
            y=edge_labels,  # Reverse x and y axis assignments
            x=edge_values,  # Reverse x and y axis assignments
            orientation='h',  # Specify horizontal orientation
            marker=dict(color='blue')
        )

        layout = go.Layout(
            title='Edge Weights Bar Chart',
            xaxis=dict(title='Weight'),  # Adjust x-axis label
            yaxis=dict(title='Edges'),  # Adjust y-axis label
            margin=dict(l=40, r=40, t=40, b=140),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace], layout=layout)
        return fig
    

    # Streamlit app
    st.title("Combined Conversation Graph and Edge Weights Analysis")

    # Extract all unique math levels
    all_math_levels = sorted(set([conversation['math_level'] for conversation in dataset]))

    # Dropdown to select math levels
    selected_math_levels = st.multiselect("Select Math Levels to Display", all_math_levels, default=all_math_levels)

    if selected_math_levels:
        # Filter dataset based on selected math levels
        filtered_dataset = [conversation for conversation in dataset if conversation['math_level'] in selected_math_levels]

        if filtered_dataset:
            st.subheader("Graph Visualization")

            # Plot combined conversation graph
            G, edge_weights = plot_combined_conversation_graph(filtered_dataset)
            st.pyplot(plt.gcf())  # Display the matplotlib figure

            st.subheader("Interactive Edge Weights Bar Chart")
            fig = plot_edge_weights_bar_chart(edge_weights)
            st.plotly_chart(fig)  # Display the Plotly interactive bar chart
        else:
            st.write("No conversations found for the selected math levels.")
    else:
        st.write("Please select at least one math level.")

# Display the Streamlit app
