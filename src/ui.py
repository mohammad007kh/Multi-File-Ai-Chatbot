import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .config import SYSTEM_MESSAGE

def setup_page():
    st.set_page_config(page_title="🧠 Multi-File Chatbot with Memory")
    st.title("🧠 Multi-File Chatbot with Memory")
    with st.expander("ℹ️ About this App"):
        st.markdown("""
Upload PDFs, Word docs, or images (OCR via OCR.space) and chat about their content.  
✅ Files & chats are never stored (memory is in RAM only, wiped on restart).
""")

def setup_session_state():
    st.session_state.setdefault("loaded_files", [])
    st.session_state.setdefault("graph", None)
    st.session_state.setdefault("thread_id", "default")  # all users share a session in Streamlit
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            SystemMessage(content=SYSTEM_MESSAGE)
        ]

def display_file_previews(extracted_texts):
    for filename, text in extracted_texts.items():
        with st.expander(f"Extracted text from: {filename}"):
            if text.strip():
                st.text_area("Text preview", text, height=200)
            else:
                st.write("_No text extracted or OCR failed._")

def display_indexed_preview(split_texts, metadatas):
    st.write("Indexed content preview:")
    for i, (chunk, meta) in enumerate(zip(split_texts, metadatas)):
        st.markdown(f"**{meta['source']} - Chunk {meta['chunk']+1}:**")
        st.code(chunk, language="markdown")
        if meta["summary"]:
            st.caption(f"Summary: {meta['summary']}")

def display_chat_history():
    for msg in st.session_state["messages"][1:]:  # Skip system message
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

def handle_chat_input(graph):
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