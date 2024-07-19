import streamlit as st
from conv_graph import show_page1
from comb_graph import show_page2
from ans_accepted import show_page3
from text_analysis import show_page4

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Individual Graphs", "Combined Graph", "Answer Accepted/Not Accpected", "Text Analysis"])

# Display the selected page
if page == "Individual Graphs":
    show_page1()
elif page == "Combined Graph":
    show_page2()
elif page == "Answer Accepted/Not Accpected":
    show_page3()
#elif page == "Text Analysis":
    #show_page4()
