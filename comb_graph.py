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

'''import streamlit as st
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
'''

import json
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go

# Load the dataset
with open('CoMTA_dataset.json') as f:
    dataset = json.load(f)

def create_conversation_graph(dataset):
    G = nx.DiGraph()
    node_freq = {}
    base_node_size = 2000

    for conversation in dataset:
        previous_category = None

        for message in conversation['data']:
            current_category = message['category']
            if current_category in node_freq:
                node_freq[current_category] += 1
            else:
                node_freq[current_category] = 1
            if previous_category:
                if G.has_edge(previous_category, current_category):
                    G[previous_category][current_category]['weight'] += 1
                else:
                    G.add_edge(previous_category, current_category, weight=1)
            previous_category = current_category

    # Add nodes with sizes based on frequency
    for node, freq in node_freq.items():
        node_size = base_node_size + 50 * freq  # Adjust node size based on frequency
        G.add_node(node, size=node_size)

    return G

def plot_weighted_graph_difference(graph1, graph2, title='Weighted Graph Difference'):
    G_diff = nx.DiGraph()
    all_nodes = set(graph1.nodes()).union(set(graph2.nodes()))

    for node in all_nodes:
        size1 = graph1.nodes[node]['size'] if node in graph1.nodes() else 0
        size2 = graph2.nodes[node]['size'] if node in graph2.nodes() else 0
        G_diff.add_node(node, size=max(size1, size2))

    for edge in set(graph1.edges()).union(set(graph2.edges())):
        weight1 = graph1[edge[0]][edge[1]]['weight'] if edge in graph1.edges() else 0
        weight2 = graph2[edge[0]][edge[1]]['weight'] if edge in graph2.edges() else 0
        diff_weight = weight1 - weight2
        if diff_weight != 0:
            G_diff.add_edge(edge[0], edge[1], weight=diff_weight)

    pos = nx.spring_layout(G_diff, k=5, iterations=50)
    node_sizes = [data['size'] for _, data in G_diff.nodes(data=True)]
    edge_weights = nx.get_edge_attributes(G_diff, 'weight')
    edge_colors = ['red' if w > 0 else 'blue' for w in edge_weights.values()]

    plt.figure(figsize=(14, 10))
    nx.draw(G_diff, pos, with_labels=True, labels={node: node for node in G_diff.nodes()}, node_size=node_sizes, node_color='skyblue', font_size=10, font_weight='bold', arrows=True, edge_color=edge_colors, width=[max(1, min(5, abs(w))) for w in edge_weights.values()])
    nx.draw_networkx_edge_labels(G_diff, pos, edge_labels=edge_weights)
    plt.title(title)

    return G_diff, edge_weights

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

def plot_combined_conversation_graph(dataset):
    G = nx.DiGraph()
    node_freq = {}
    base_node_size = 2000

    for conversation in dataset:
        previous_category = None

        for message in conversation['data']:
            current_category = message['category']
            if current_category in node_freq:
                node_freq[current_category] += 1
            else:
                node_freq[current_category] = 1
            if previous_category:
                if G.has_edge(previous_category, current_category):
                    G[previous_category][current_category]['weight'] += 1
                else:
                    G.add_edge(previous_category, current_category, weight=1)
            previous_category = current_category

    # Add nodes with sizes based on frequency
    for node, freq in node_freq.items():
        node_size = base_node_size + 50 * freq  # Adjust node size based on frequency
        G.add_node(node, size=node_size)

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
            st.subheader("Graph Visualization")

            # Plot combined conversation graph for all selected math levels
            st.subheader("Combined Adjacency Graph for Selected Math Levels")
            G_combined, edge_weights_combined = plot_combined_conversation_graph(filtered_dataset)
            st.pyplot(plt.gcf())

            st.subheader("Interactive Edge Weights Bar Chart for Combined Graph")
            fig_combined = plot_edge_weights_bar_chart(edge_weights_combined)
            st.plotly_chart(fig_combined)  # Display the Plotly interactive bar chart

            # Plot individual conversation graphs for each selected math level
            for math_level in selected_math_levels:
                st.subheader(f"Combined Adjacency Graph for {math_level}")
                G_math_level = create_conversation_graph([conversation for conversation in filtered_dataset if conversation['math_level'] == math_level])
                pos = nx.spring_layout(G_math_level, k=5, iterations=50)
                node_sizes = [data['size'] for _, data in G_math_level.nodes(data=True)]
                edge_weights = nx.get_edge_attributes(G_math_level, 'weight')

                plt.figure(figsize=(14, 10))
                nx.draw(G_math_level, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
                nx.draw_networkx_edge_labels(G_math_level, pos, edge_labels=edge_weights)
                st.pyplot(plt.gcf())

                st.subheader(f"Interactive Edge Weights Bar Chart for {math_level}")
                fig_math_level = plot_edge_weights_bar_chart(edge_weights)
                st.plotly_chart(fig_math_level)

            # Compare two selected math levels
            st.subheader("Select Two Math Levels to Compare Their Weighted Graphs")
            selected_comparison_levels = st.multiselect("Select Two Math Levels for Comparison", selected_math_levels)

            if len(selected_comparison_levels) == 2:
                math_level_1, math_level_2 = selected_comparison_levels

                # Create conversation graphs for selected math levels
                G_math_level_1 = create_conversation_graph([conversation for conversation in filtered_dataset if conversation['math_level'] == math_level_1])
                G_math_level_2 = create_conversation_graph([conversation for conversation in filtered_dataset if conversation['math_level'] == math_level_2])

                # Plot weighted graph difference between two math levels
                st.subheader(f"Weighted Graph Difference Between {math_level_1} and {math_level_2}")
                G_diff, edge_weights_diff = plot_weighted_graph_difference(G_math_level_1, G_math_level_2, title=f"{math_level_1} vs {math_level_2} Graph Difference")
                st.pyplot(plt.gcf())

                # Interactive Edge Weights Bar Chart for Graph Difference
                st.subheader("Interactive Edge Weights Bar Chart for Graph Difference")
                fig_diff = plot_edge_weights_bar_chart(edge_weights_diff)
                st.plotly_chart(fig_diff)  # Display the Plotly interactive bar chart

            elif len(selected_comparison_levels) > 2:
                st.write("Please select exactly two math levels for comparison.")

            # Compare a specific math level against all others
            st.subheader("Select a Math Level to Compare Against All Others")
            primary_math_level = st.selectbox("Select Primary Math Level", selected_math_levels)

            if primary_math_level:
                other_math_levels = [level for level in selected_math_levels if level != primary_math_level]

                # Create conversation graphs
                G_primary = create_conversation_graph([conversation for conversation in filtered_dataset if conversation['math_level'] == primary_math_level])
                G_others = create_conversation_graph([conversation for conversation in filtered_dataset if conversation['math_level'] in other_math_levels])

                # Plot weighted graph difference between primary math level and all others
                st.subheader(f"Weighted Graph Difference Between {primary_math_level} and All Other Selected Math Levels")
                G_diff_primary, edge_weights_diff_primary = plot_weighted_graph_difference(G_primary, G_others, title=f"{primary_math_level} vs All Others Graph Difference")
                st.pyplot(plt.gcf())

                # Interactive Edge Weights Bar Chart for Graph Difference
                st.subheader("Interactive Edge Weights Bar Chart for Graph Difference")
                fig_diff_primary = plot_edge_weights_bar_chart(edge_weights_diff_primary)
                st.plotly_chart(fig_diff_primary)  # Display the Plotly interactive bar chart

        else:
            st.write("No conversations found for the selected math levels.")
    else:
        st.write("Please select at least one math level.")
