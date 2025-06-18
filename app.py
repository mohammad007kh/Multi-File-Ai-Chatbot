import streamlit as st
import requests
import openai
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="🧠 Multi-File Chatbot with Memory")
st.title("🧠 Multi-File Chatbot with Memory")
with st.expander("ℹ️ About this App"):
    st.markdown("""
Upload PDFs, Word docs, or images (OCR via OCR.space) and chat about their content.
✅ Files & chats are never stored.
""")

# --- Load secrets ---
openai_key = st.secrets["openai_api_key"]
ocr_space_key = st.secrets["ocr_space_api_key"]

# --- Upload section ---
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Session state ---
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("loaded_files", [])
st.session_state.setdefault("qa_chain", None)

# --- OCR helper ---
def ocr_space_text(img_bytes):
    try:
        payload = {"apikey": ocr_space_key, "language": "eng"}
        files = {"file": img_bytes}
        r = requests.post("https://api.ocr.space/parse/image", files=files, data=payload, timeout=60)
        r.raise_for_status()
        result = r.json()
        return "\n".join(item["ParsedText"].strip() for item in result.get("ParsedResults", []))
    except requests.exceptions.RequestException as e:
        st.error(f"OCR request failed: {e}")
        return ""

# --- Text extraction ---
def extract_text(f):
    if f.name.lower().endswith(".pdf"):
        reader = PdfReader(f)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if f.name.lower().endswith(".docx"):
        doc = docx.Document(f)
        return "\n".join(p.text for p in doc.paragraphs)
    if f.name.lower().endswith((".png", "jpg", "jpeg")):
        return ocr_space_text(f)
    return ""

# --- Use OpenAI to describe document ---
def describe_document(text, filename):
    client = OpenAI(api_key=openai_key)
    prompt = f"Describe the content of the following document named '{filename}'. Give a short summary and possible categories:\n\n{text[:2000]}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Failed to describe document: {e}")
        return ""

# --- Update vectorstore when files uploaded ---
if uploaded:
    names = [f.name for f in uploaded]
    if set(names) != set(st.session_state.loaded_files):
        extracted_texts = {}
        for f in uploaded:
            text = extract_text(f)
            extracted_texts[f.name] = text

        # Display previews
        for filename, text in extracted_texts.items():
            with st.expander(f"Extracted text from: {filename}"):
                if text.strip():
                    st.text_area("Text preview", text, height=200)
                else:
                    st.write("_No text extracted or OCR failed._")

        # Split text and add metadata
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_texts, metadatas = [], []

        for filename, doc_text in extracted_texts.items():
            chunks = splitter.split_text(doc_text)
            summary = describe_document(doc_text, filename)
            for chunk in chunks:
                split_texts.append(chunk)
                metadatas.append({
                    "source": filename,
                    "summary": summary
                })

        # Vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vs = FAISS.from_texts(split_texts, embeddings, metadatas=metadatas)

        # Memory + chain
        mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(api_key=openai_key, temperature=0),
            retriever=vs.as_retriever(search_kwargs={"k": 5}),
            memory=mem,
            return_source_documents=True
        )

        st.session_state.update(qa_chain=qa, memory=mem, chat_history=[], loaded_files=names)

        # Show indexed preview
        st.write("Indexed content preview:")
        for i, (chunk, meta) in enumerate(zip(split_texts, metadatas)):
            st.markdown(f"**{meta['source']} - Chunk {i+1}:**")
            st.code(chunk, language="markdown")
            st.caption(f"Summary: {meta['summary']}")

# --- Chat Interface ---
qa = st.session_state.qa_chain
if qa:
    query = st.chat_input("Ask a question…")
    if query:
        res = qa({"question": query})
        ans = "I don’t know." if not res["source_documents"] else res["answer"]
        st.session_state.chat_history += [("user", query), ("assistant", ans)]

# --- Display chat history ---
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)
