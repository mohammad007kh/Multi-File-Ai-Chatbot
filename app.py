import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

import streamlit as st
from tempfile import NamedTemporaryFile

# Set API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("📄 PDF RAG Chatbot (Guarded Responses)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            memory.chat_memory.add_message(HumanMessage(content=msg))
        elif speaker == "Bot":
            memory.chat_memory.add_message(AIMessage(content=msg))

    # LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # 🔐 Custom prompt for grounded answers
    RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use only the information provided in the context to answer the user's question.

If you do not know the answer based on the context, simply respond: "I don't know."

Context:
{context}

Chat History:
{chat_history}

User: {question}
Assistant:
"""
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # RAG Chain with guarded behavior
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    # Chat
    user_input = st.text_input("Ask a question about the PDF:")
    if user_input:
        response = qa({"question": user_input})
        answer = response["answer"]

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
