import streamlit as st
import requests
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx
from PIL import Image

# LangGraph imports
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Page setup ---
st.set_page_config(page_title="🧠 Multi-File Chatbot with Memory")
st.title("🧠 Multi-File Chatbot with Memory")
with st.expander("ℹ️ About this App"):
    st.markdown("""
Upload PDFs, Word docs, or images (OCR via OCR.space) and chat about their content.  
✅ Files & chats are never stored (memory is in RAM only, wiped on restart).
""")

# --- Load secrets ---
openai_key = st.secrets["openai_api_key"]
ocr_space_key = st.secrets["ocr_space_api_key"]

# --- Upload section ---
uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

## --- Session state ---
st.session_state.setdefault("loaded_files", [])
st.session_state.setdefault("graph", None)
st.session_state.setdefault("thread_id", "default")  # all users share a session in Streamlit
# Store all chat messages for memory (role-based dicts)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(content="You are a helpful assistant. Answer questions based on the user's uploaded files and previous conversation.")
    ]

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
        return "\n".join(p.extract_text() or "" for p in reader.pages), False
    if f.name.lower().endswith(".docx"):
        doc = docx.Document(f)
        return "\n".join(p.text for p in doc.paragraphs), False
    if f.name.lower().endswith((".png", "jpg", "jpeg")):
        return ocr_space_text(f), True
    return "", False

# --- Use OpenAI to describe document ---
def describe_document(text, filename):
    client = OpenAI(api_key=openai_key)
    prompt = f"Describe the content of the following document named '{filename}'. Give a short summary and possible categories:\n\n{text[:1500]}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        ocr_flags = {}

        for f in uploaded:
            text, is_ocr = extract_text(f)
            extracted_texts[f.name] = text
            ocr_flags[f.name] = is_ocr

        # Display previews
        for filename, text in extracted_texts.items():
            with st.expander(f"Extracted text from: {filename}"):
                if text.strip():
                    st.text_area("Text preview", text, height=200)
                else:
                    st.write("_No text extracted or OCR failed._")

        # Split text and add metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_texts, metadatas = [], []

        for filename, doc_text in extracted_texts.items():
            chunks = splitter.split_text(doc_text)
            summary = describe_document(doc_text, filename) if ocr_flags.get(filename) else ""
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                split_texts.append(chunk)
                metadatas.append({
                    "source": filename,
                    "summary": summary,
                    "chunk": i
                })

        # Vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vs = FAISS.from_texts(split_texts, embeddings, metadatas=metadatas)

        # --- LangGraph setup ---
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

        def chatbot(state: MessagesState):
            """One chatbot step: take messages, run LLM with retrieval"""
            messages = state["messages"]
            question = messages[-1].content
            retriever = vs.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join(d.page_content for d in docs)
            
            # Create a new prompt that includes context and passes all messages to LLM
            context_message = f"Document context:\n{context}\n\n"
            
            # Create messages for the LLM including context
            llm_messages = [messages[0]]  # System message
            llm_messages.append(HumanMessage(content=context_message + "Now I'll share our conversation:"))
            llm_messages.extend(messages[1:])  # Add all conversation messages
            
            answer = llm.invoke(llm_messages)
            return {"messages": [answer]}

        builder = StateGraph(MessagesState)
        builder.add_node("chatbot", chatbot)
        builder.set_entry_point("chatbot")
        builder.add_edge("chatbot", END)

        # In-memory only (no persistence)
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        st.session_state.update(graph=graph, loaded_files=names)

        # Show indexed preview
        st.write("Indexed content preview:")
        for i, (chunk, meta) in enumerate(zip(split_texts, metadatas)):
            st.markdown(f"**{meta['source']} - Chunk {meta['chunk']+1}:**")
            st.code(chunk, language="markdown")
            if meta["summary"]:
                st.caption(f"Summary: {meta['summary']}")

# --- Chat Interface ---
graph = st.session_state.graph
if graph:
    # Display previous chat history
    for msg in st.session_state["messages"][1:]:  # Skip system message
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
    
    query = st.chat_input("Ask a question…")
    if query:
        # Add user message to history as LangChain message
        st.session_state["messages"].append(HumanMessage(content=query))
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        # Pass full message history to graph
        res = graph.invoke({"messages": st.session_state["messages"]}, config=config)
        answer = res["messages"][-1].content
        st.session_state["messages"].append(AIMessage(content=answer))
        st.rerun()  # Refresh to show the new messages