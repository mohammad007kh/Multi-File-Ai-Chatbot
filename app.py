import streamlit as st
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
from PIL import Image
import base64

st.set_page_config(page_title="🧠 Multi-File Chatbot with Memory")
st.title("🧠 Multi-File Chatbot with Memory")

with st.expander("ℹ️ About this App"):
    st.markdown("""
Upload PDFs, Word docs, or images (OCR via OCR.space) and chat about their content.
✅ Files & chats are never stored.
""")

# Load secrets
openai_key = st.secrets["openai_api_key"]
ocr_space_key = st.secrets["ocr_space_api_key"]

# Upload
uploaded = st.file_uploader("Upload files", type=["pdf","docx","png","jpg","jpeg"], accept_multiple_files=True)

# Session state
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("loaded_files", [])
st.session_state.setdefault("qa_chain", None)

# OCR.space helper
def ocr_space_text(img_bytes):
    payload = {"apikey": ocr_space_key, "language": "eng"}
    files = {"file": img_bytes}
    r = requests.post("https://api.ocr.space/parse/image", files=files, data=payload)
    result = r.json()
    return "\n".join(item["ParsedText"].strip() for item in result.get("ParsedResults", []))

# Extract text
def extract_text(f):
    if f.name.lower().endswith(".pdf"):
        reader = PdfReader(f); return "\n".join(p.extract_text() or "" for p in reader.pages)
    if f.name.lower().endswith(".docx"):
        doc = docx.Document(f); return "\n".join(p.text for p in doc.paragraphs)
    if f.name.lower().endswith((".png","jpg","jpeg")):
        return ocr_space_text(f)
    return ""

# Update vectorstore
if uploaded:
    names = [f.name for f in uploaded]
    if set(names) != set(st.session_state.loaded_files):
        text = "\n".join(extract_text(f) for f in uploaded)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vs = FAISS.from_texts([text], embeddings)
        mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(api_key=openai_key, temperature=0),
            retriever=vs.as_retriever(search_kwargs={"k":3}),
            memory=mem,
            return_source_documents=True
        )
        st.session_state.update(qa_chain=qa, memory=mem, chat_history=[], loaded_files=names)

# Chat interface
qa = st.session_state.qa_chain
if qa:
    query = st.chat_input("Ask a question…")
    if query:
        res = qa({"question": query})
        ans = "I don’t know." if not res["source_documents"] else res["answer"]
        st.session_state.chat_history += [("user", query), ("assistant", ans)]

# Display
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)
