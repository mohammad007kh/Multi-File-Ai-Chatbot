import streamlit as st
from src.ui import setup_page, setup_session_state, display_file_previews, display_indexed_preview, display_chat_history, handle_chat_input
from src.vectorstore import create_vectorstore
from src.chatbot import create_chatbot_graph

# --- Page setup ---
setup_page()

# --- Upload section ---
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Session state ---
setup_session_state()

# --- Update vectorstore when files uploaded ---
if uploaded:
    names = [f.name for f in uploaded]
    if set(names) != set(st.session_state.loaded_files):
        vs, extracted_texts, ocr_flags, split_texts, metadatas = create_vectorstore(uploaded)

        # Display previews
        display_file_previews(extracted_texts)

        # Create chatbot graph
        graph = create_chatbot_graph(vs)

        st.session_state.update(graph=graph, loaded_files=names)

        # Show indexed preview
        display_indexed_preview(split_texts, metadatas)

# --- Chat Interface ---
graph = st.session_state.graph
if graph:
    display_chat_history()
    handle_chat_input(graph)