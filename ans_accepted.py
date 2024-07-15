'''import json
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go

# Load dataset
with open('CoMTA_dataset.json') as f:
    dataset = json.load(f)

def show_page3():
    # Function to plot conversation graph and return edge weights
    def plot_conversation_graph(conversations, title):
        G = nx.DiGraph()
        edge_weights = {}

        for conversation in conversations:
            previous_category = None
            for message in conversation['data']:
                current_category = message['category']
                G.add_node(current_category, size=3000 + 100 * len(current_category))
                if previous_category:
                    if G.has_edge(previous_category, current_category):
                        G[previous_category][current_category]['weight'] += 1
                    else:
                        G.add_edge(previous_category, current_category, weight=1)
                previous_category = current_category

        positions = nx.spring_layout(G, k=5, iterations=50)  # Using spring layout with adjusted parameters

        edge_weights = nx.get_edge_attributes(G, 'weight')

        fig, ax = plt.subplots(figsize=(14, 10))  # Create a figure and axis object
        nx.draw(G, positions, with_labels=True, labels={node: node for node in G.nodes()}, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True, ax=ax)
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_weights, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)  # Display the Matplotlib figure in Streamlit

        return edge_weights  # Return edge weights dictionary

    # Function to create interactive bar chart
    def plot_interactive_bar_chart(edge_weights, chart_title):
        edge_labels = [' -> '.join(edge) for edge in edge_weights.keys()]
        edge_values = list(edge_weights.values())

        trace = go.Bar(
            y=edge_labels,  # Reverse x and y axis assignments
            x=edge_values,  # Reverse x and y axis assignments
            orientation='h',  # Specify horizontal orientation
            marker=dict(color='blue')
        )

        layout = go.Layout(
            title=chart_title,
            xaxis=dict(title='Weight'),  # Adjust x-axis label
            yaxis=dict(title='Edges'),  # Adjust y-axis label
            margin=dict(l=40, r=40, t=40, b=140),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig)  # Display the Plotly interactive bar chart in Streamlit

    # Separate conversations based on the expected_result
    accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Accepted']
    not_accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Not Accepted']

    # Streamlit app
    st.title("Conversation Analysis")

    st.subheader("Conversation Graph for Answer Accepted")
    edge_weights_accepted = plot_conversation_graph(accepted_conversations, 'Conversation Graph for Answer Accepted')

    st.subheader("Interactive Edge Weights Bar Chart for Answer Accepted")
    plot_interactive_bar_chart(edge_weights_accepted, 'Edge Weights Bar Chart for Answer Accepted')

    st.subheader("Conversation Graph for Answer Not Accepted")
    edge_weights_not_accepted = plot_conversation_graph(not_accepted_conversations, 'Conversation Graph for Answer Not Accepted')

    st.subheader("Interactive Edge Weights Bar Chart for Answer Not Accepted")
    plot_interactive_bar_chart(edge_weights_not_accepted, 'Edge Weights Bar Chart for Answer Not Accepted')
'''

import json
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go

# Load dataset
with open('CoMTA_dataset.json') as f:
    dataset = json.load(f)

def show_page3():
    # Function to plot conversation graph and return edge weights
    def plot_conversation_graph(conversations, title):
        G = nx.DiGraph()
        edge_weights = {}
        node_frequency = {}

        # Calculate node frequency
        for conversation in conversations:
            for message in conversation['data']:
                current_category = message['category']
                if current_category in node_frequency:
                    node_frequency[current_category] += 1
                else:
                    node_frequency[current_category] = 1

        # Add nodes and edges
        for conversation in conversations:
            previous_category = None
            for message in conversation['data']:
                current_category = message['category']
                #base_size = 300  # Base node size
                size = 40 * node_frequency[current_category]
                G.add_node(current_category, size=size)
                if previous_category:
                    if G.has_edge(previous_category, current_category):
                        G[previous_category][current_category]['weight'] += 1
                    else:
                        G.add_edge(previous_category, current_category, weight=1)
                previous_category = current_category

        positions = nx.spring_layout(G, k=5, iterations=50)  # Using spring layout with adjusted parameters

        edge_weights = nx.get_edge_attributes(G, 'weight')
        node_sizes = [G.nodes[node]['size'] for node in G.nodes]

        fig, ax = plt.subplots(figsize=(14, 10))  # Create a figure and axis object
        nx.draw(G, positions, with_labels=True, labels={node: node for node in G.nodes()},
                node_size=node_sizes, node_color='skyblue', font_size=10, font_weight='bold', arrows=True, ax=ax)
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_weights, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)  # Display the Matplotlib figure in Streamlit

        return edge_weights  # Return edge weights dictionary

    # Function to create interactive bar chart
    def plot_interactive_bar_chart(edge_weights, chart_title):
        edge_labels = [' -> '.join(edge) for edge in edge_weights.keys()]
        edge_values = list(edge_weights.values())

        trace = go.Bar(
            y=edge_labels,  # Reverse x and y axis assignments
            x=edge_values,  # Reverse x and y axis assignments
            orientation='h',  # Specify horizontal orientation
            marker=dict(color='blue')
        )

        layout = go.Layout(
            title=chart_title,
            xaxis=dict(title='Weight'),  # Adjust x-axis label
            yaxis=dict(title='Edges'),  # Adjust y-axis label
            margin=dict(l=40, r=40, t=40, b=140),
            hovermode='closest'
        )

        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig)  # Display the Plotly interactive bar chart in Streamlit

    # Separate conversations based on the expected_result and math_level
    math_levels = set(conv['math_level'] for conv in dataset)

    # Streamlit app
    st.title("Conversation Analysis")

    st.header("Combined Analysis")

    accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Accepted']
    not_accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Not Accepted']

    st.subheader("Combined Conversation Graph for Answer Accepted")
    edge_weights_accepted = plot_conversation_graph(accepted_conversations, 'Combined Conversation Graph for Answer Accepted')

    st.subheader("Combined Interactive Edge Weights Bar Chart for Answer Accepted")
    plot_interactive_bar_chart(edge_weights_accepted, 'Combined Edge Weights Bar Chart for Answer Accepted')

    st.subheader("Combined Conversation Graph for Answer Not Accepted")
    edge_weights_not_accepted = plot_conversation_graph(not_accepted_conversations, 'Combined Conversation Graph for Answer Not Accepted')

    st.subheader("Combined Interactive Edge Weights Bar Chart for Answer Not Accepted")
    plot_interactive_bar_chart(edge_weights_not_accepted, 'Combined Edge Weights Bar Chart for Answer Not Accepted')

    for level in math_levels:
        st.header(f"Math Level: {level}")

        accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Accepted' and conv['math_level'] == level]
        not_accepted_conversations = [conv for conv in dataset if conv['expected_result'] == 'Answer Not Accepted' and conv['math_level'] == level]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Conversation Graph for Answer Accepted")
            edge_weights_accepted = plot_conversation_graph(accepted_conversations, f'Conversation Graph for Answer Accepted - {level}')

            st.subheader("Interactive Edge Weights Bar Chart for Answer Accepted")
            plot_interactive_bar_chart(edge_weights_accepted, f'Edge Weights Bar Chart for Answer Accepted - {level}')

        with col2:
            st.subheader("Conversation Graph for Answer Not Accepted")
            edge_weights_not_accepted = plot_conversation_graph(not_accepted_conversations, f'Conversation Graph for Answer Not Accepted - {level}')

            st.subheader("Interactive Edge Weights Bar Chart for Answer Not Accepted")
            plot_interactive_bar_chart(edge_weights_not_accepted, f'Edge Weights Bar Chart for Answer Not Accepted - {level}')

# Display the Streamlit app
if __name__ == '__main__':
    show_page3()


