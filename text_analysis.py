import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
import nltk
import re
import streamlit as st
# Download the stopwords from NLTK
nltk.download('stopwords')

# Define stopwords and additional unwanted words
stop_words = set(stopwords.words('english'))
additional_stopwords = {"now", "lets", "let's", "see", "let"}
unwanted_words = stop_words.union(additional_stopwords)

# Function to tokenize and clean words
def tokenize(text):
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())  # Keep only words with 2 or more characters
    filtered_words = [word for word in words if word not in unwanted_words]
    return filtered_words

# Load the dataset
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to remove stop words and symbols
def remove_stop_words_and_symbols(message):
    words = re.findall(r'\b[a-zA-Z]+\b', message.lower())
    filtered_words = [word for word in words if word not in unwanted_words]
    return filtered_words

# Function to concatenate replies
def concatenate_replies(data, role):
    replies = [exchange['content'] for exchange in data if exchange['role'] == role]
    filtered_replies = []
    for reply in replies:
        filtered_replies.extend(remove_stop_words_and_symbols(reply))
    return ' '.join(filtered_replies)

# Streamlit app
def main():
    st.title('Conversation Dataset Analysis')

    # Load dataset
    file_path = st.text_input("Enter the path to the JSON file:", 'CoMTA_dataset.json')
    if file_path:
        dataset = load_data(file_path)

        # User and Assistant Replies
        user_replies = ' '.join([concatenate_replies(entry['data'], 'user') for entry in dataset])
        assistant_replies = ' '.join([concatenate_replies(entry['data'], 'assistant') for entry in dataset])

        # Word Clouds
        st.subheader('Word Clouds')
        
        user_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_replies)
        st.image(user_wordcloud.to_array(), caption='Word Cloud for User Replies')
        
        assistant_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(assistant_replies)
        st.image(assistant_wordcloud.to_array(), caption='Word Cloud for Assistant Replies')

        # Calculate the length of each conversation
        conversation_lengths = [len(conv['data']) for conv in dataset]
        df = pd.DataFrame([{'data': conv['data'], 'math_level': conv['math_level']} for conv in dataset])
        df['conversation_length'] = conversation_lengths

        # Create an interactive histogram for conversation lengths
        fig = px.histogram(conversation_lengths, 
                        nbins=max(conversation_lengths) - 1, 
                        labels={'x': 'Number of Messages', 'count': 'Number of Conversations'},
                        title='Distribution of Conversation Lengths')
        fig.update_layout(xaxis_title='Number of Messages',
                          yaxis_title='Number of Conversations',
                          bargap=0.2)  # Adjust the gap between bars for clarity
        st.plotly_chart(fig)

        # Average Length of Conversations
        average_length = sum(conversation_lengths) / len(conversation_lengths)
        st.write(f"Average Length of Conversations: {average_length:.2f} exchanges")

        # Scatter plot of conversation length vs. math level
        st.subheader('Scatter Plot of Conversation Length vs. Math Level')

        fig = px.scatter(df, x='math_level', y='conversation_length',
                         labels={'math_level': 'Math Level', 'conversation_length': 'Conversation Length (Number of Exchanges)'},
                         title='Scatter Plot of Conversation Length vs. Math Level')
        fig.update_layout(
            xaxis_title='Math Level',
            yaxis_title='Conversation Length (Number of Exchanges)',
            title='Scatter Plot of Conversation Length vs. Math Level'
        )

        st.plotly_chart(fig)

        # Text Lengths Distribution
        user_text_lengths = [len(concatenate_replies(entry['data'], 'user')) for entry in dataset]
        assistant_text_lengths = [len(concatenate_replies(entry['data'], 'assistant')) for entry in dataset]

        st.subheader('Text Lengths Distribution')
        
        fig = px.histogram(user_text_lengths, nbins=20, title='Distribution of Text Lengths (User Replies)', 
                           labels={'x': 'Text Length (Number of Characters)', 'count': 'Frequency'})
        fig.update_layout(xaxis_title='Text Length (Number of Characters)',
                          yaxis_title='Frequency',
                          bargap=0.2)  # Adjust bar gap
        st.plotly_chart(fig)
        
        fig = px.histogram(assistant_text_lengths, nbins=20, title='Distribution of Text Lengths (Assistant Replies)', 
                           labels={'x': 'Text Length (Number of Characters)', 'count': 'Frequency'})
        fig.update_layout(xaxis_title='Text Length (Number of Characters)',
                          yaxis_title='Frequency',
                          bargap=0.2)  # Adjust bar gap
        st.plotly_chart(fig)

        # Words per Chat
        words_per_chat = [sum(len(entry['content'].split()) for entry in conv['data']) for conv in dataset]
        
        st.subheader('Words Per Chat')
        
        fig = px.histogram(words_per_chat, nbins=max(words_per_chat) // 50, 
                           labels={'x': 'Number of Words', 'count': 'Number of Chats'},
                           title='Distribution of Words per Chat')
        fig.update_layout(xaxis_title='Number of Words',
                          yaxis_title='Number of Chats',
                          bargap=0.2)  # Adjust bar gap
        st.plotly_chart(fig)

        # Detailed Statistics
        words_per_chat_stats = {
            'min_words': min(words_per_chat),
            'max_words': max(words_per_chat),
            'average_words': sum(words_per_chat) / len(words_per_chat)
        }

        st.write("Words Per Chat Statistics:", words_per_chat_stats)
        
        # AI vs. Human Words
        ai_words = []
        human_words = []
        
        for conv in dataset:
            ai_count = 0
            human_count = 0
            for entry in conv['data']:
                word_count = len(entry['content'].split())
                if entry['role'] == 'assistant':
                    ai_count += word_count
                else:
                    human_count += word_count
            ai_words.append(ai_count)
            human_words.append(human_count)
        
        st.subheader('AI vs. Human Words')
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ai_words, name='AI', opacity=0.5))
        fig.add_trace(go.Histogram(x=human_words, name='Human', opacity=0.5))
        fig.update_layout(title='Distribution of Words by AI vs. Human',
                          xaxis_title='Number of Words',
                          yaxis_title='Number of Chats',
                          barmode='overlay',
                          bargap=0.2)  # Adjust bar gap
        st.plotly_chart(fig)

        # Detailed Statistics for AI and Human Words
        ai_words_stats = {
            'min_words': min(ai_words),
            'max_words': max(ai_words),
            'average_words': sum(ai_words) / len(ai_words)
        }
        
        human_words_stats = {
            'min_words': min(human_words),
            'max_words': max(human_words),
            'average_words': sum(human_words) / len(human_words)
        }
        
        st.write("AI Words Statistics:", ai_words_stats)
        st.write("Human Words Statistics:", human_words_stats)

        # Average Words Per Message
        ai_message_counts = []
        human_message_counts = []

        for conv in dataset:
            ai_messages = 0
            human_messages = 0
            ai_word_count = 0
            human_word_count = 0
            for entry in conv['data']:
                word_count = len(entry['content'].split())
                if entry['role'] == 'assistant':
                    ai_messages += 1
                    ai_word_count += word_count
                else:
                    human_messages += 1
                    human_word_count += word_count
            if ai_messages > 0:
                ai_message_counts.append(ai_word_count / ai_messages)
            if human_messages > 0:
                human_message_counts.append(human_word_count / human_messages)
        
        st.subheader('Average Words per Message for AI vs. Human')
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ai_message_counts, name='AI', opacity=0.5))
        fig.add_trace(go.Histogram(x=human_message_counts, name='Human', opacity=0.5))
        fig.update_layout(title='Average Words per Message for AI vs. Human',
                          xaxis_title='Average Number of Words per Message',
                          yaxis_title='Number of Chats',
                          barmode='overlay',
                          bargap=0.2)  # Adjust bar gap
        st.plotly_chart(fig)

        # Most Common Words
        ai_word_counter = Counter()
        human_word_counter = Counter()
        
        for conv in dataset:
            for entry in conv['data']:
                words = tokenize(entry['content'])
                if entry['role'] == 'assistant':
                    ai_word_counter.update(words)
                else:
                    human_word_counter.update(words)

        most_common_ai_words = ai_word_counter.most_common(10)
        most_common_human_words = human_word_counter.most_common(10)
        unique_ai_words = len(ai_word_counter)
        unique_human_words = len(human_word_counter)
        
        st.subheader('Most Common Words')
        
        ai_words, ai_counts = zip(*most_common_ai_words)
        fig = px.bar(x=list(ai_counts), y=list(ai_words), labels={'x': 'Frequency', 'y': 'Words'},
                     title='Most Common AI Words', text=list(ai_counts))
        fig.update_layout(bargap=0.4)  # Adjust bar gap
        st.plotly_chart(fig)
        
        human_words, human_counts = zip(*most_common_human_words)
        fig = px.bar(x=list(human_counts), y=list(human_words), labels={'x': 'Frequency', 'y': 'Words'},
                     title='Most Common Human Words', text=list(human_counts))
        fig.update_layout(bargap=0.4)  # Adjust bar gap
        st.plotly_chart(fig)

        st.write(f"Unique AI Words: {unique_ai_words}")
        st.write(f"Unique Human Words: {unique_human_words}")

        # Average Sentence Length
        ai_sentence_lengths = []
        human_sentence_lengths = []

        for conv in dataset:
            for entry in conv['data']:
                sentence_lengths = [len(sentence.split()) for sentence in re.split(r'[.!?]', entry['content']) if sentence]
                if entry['role'] == 'assistant':
                    ai_sentence_lengths.extend(sentence_lengths)
                else:
                    human_sentence_lengths.extend(sentence_lengths)

        avg_ai_sentence_length = sum(ai_sentence_lengths) / len(ai_sentence_lengths)
        avg_human_sentence_length = sum(human_sentence_lengths) / len(human_sentence_lengths)
        
        st.write(f"Average AI Sentence Length: {avg_ai_sentence_length:.2f} words")
        st.write(f"Average Human Sentence Length: {avg_human_sentence_length:.2f} words")

def show_page4():
    main()
