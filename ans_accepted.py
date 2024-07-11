import json
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go

# Load dataset
with open('c:/Users/admin/Downloads/CoMTA_dataset.json') as f:
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
            x=edge_labels,
            y=edge_values,
            marker=dict(color='blue')
        )

        layout = go.Layout(
            title=chart_title,
            xaxis=dict(title='Edges'),
            yaxis=dict(title='Weight'),
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
