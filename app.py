import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from PyPDF2 import PdfReader
import docx
import pytesseract
from PIL import Image
import os

st.set_page_config(page_title="🧠 Multi-File Chatbot with Memory")

st.title("🧠 Multi-File Chatbot with Memory")

# Info & Disclaimer
with st.expander("ℹ️ About this App"):
    st.markdown("""
This is a demo RAG-based chatbot that can answer your questions based on uploaded PDFs, Word docs, and images.

✅ **Your files and conversations are not stored.**  
    """)

# Load API key from secrets
openai_api_key = st.secrets["openai_api_key"]

# File upload
uploaded_files = st.file_uploader("Upload PDF, DOCX, or image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# File parsers
def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif name.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    else:
        return ""

# Update vectorstore if new files
if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if set(current_names) != set(st.session_state.loaded_files):
        full_text = ""
        for file in uploaded_files:
            full_text += extract_text(file) + "\n"

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts([full_text], embedding=embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
        )


        st.session_state.qa_chain = qa_chain
        st.session_state.memory = memory
        st.session_state.chat_history = []
        st.session_state.loaded_files = current_names

# Chat interface
if st.session_state.qa_chain:
    question = st.chat_input("Ask a question about your uploaded files...")
    if question:
        result = st.session_state.qa_chain({"question": question})

        # Check if source docs are empty → fallback
        if not result["source_documents"]:
            answer = "I don’t know."
        else:
            answer = result["answer"]

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("bot", answer))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
