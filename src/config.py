import streamlit as st

# Load secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
OCR_SPACE_API_KEY = st.secrets["ocr_space_api_key"]

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 50
RETRIEVER_K = 5
SYSTEM_MESSAGE = "You are a helpful assistant. Answer questions based on the user's uploaded files and previous conversation."