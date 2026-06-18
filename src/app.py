# app.py
import streamlit as st
from chain import manual_ytLoader, create_retriever, generate_ans

# 1. Page Configuration
st.set_page_config(page_title="YouTube ChatApp", layout="centered")
st.title("📺 YouTube Video ChatApp")

# 2. Maintain Retriever across Reruns
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# 3. Sidebar UI (Indexing button("Index Video")
with st.sidebar:
    st.header("Setup")
    url_input = st.text_input("Enter YouTube URL:")
    process_button = st.button("Index Video")
    
    if process_button and url_input:
        with st.spinner("Processing Video..."):
            doc = manual_ytLoader(url_input)
            if doc:
                # Store the retriever instance inside session state
                st.session_state.retriever = create_retriever(doc)
                st.success("Video indexed successfully!")
            else:
                st.error("Could not fetch video details. Check URL or video captions.")

# 4. Main Chat Window UI (Generation Pipeline)
user_question = st.text_input("Ask a question about the video:")

if user_question:
    if st.session_state.retriever:
        with st.spinner("Thinking..."):
            # Pass query and cached retriever right into your LCEL pipeline
            response = generate_ans(user_question, st.session_state.retriever)
            
            st.write("### Answer:")
            st.write(response)
    else:
        st.warning("Please enter a YouTube URL and click 'Index Video' in the sidebar first.")
