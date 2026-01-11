from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from .config import OPENAI_API_KEY, RETRIEVER_K

def create_chatbot_graph(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    def chatbot(state: MessagesState):
        """One chatbot step: take messages, run LLM with retrieval"""
        messages = state["messages"]
        question = messages[-1].content
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        docs = retriever.invoke(question)
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

    return graph