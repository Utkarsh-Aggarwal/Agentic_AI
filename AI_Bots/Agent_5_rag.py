from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_core.tools import tool

from langchain_community.embeddings import HuggingFaceEmbeddings


# GROQ LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key="gsk_TXqZWHoxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxy"
)



# Local Embeddings (since Groq doesn't provide embeddings)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


pdf_path = "Stock_Market_Performance_2024.pdf"


# Safety check
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")


# Load PDF
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)


persist_directory = "./chroma_db"
collection_name = "stock_market"


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


# Create ChromaDB
try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    print("Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    Search and return information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

# Bind tools to LLM
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if LLM wants to call a tool"""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an AI assistant that answers questions about Stock Market Performance in 2024.

Use the retriever tool to search the PDF document if needed.

Always cite the document sections used in the answer.
"""


tools_dict = {tool.name: tool for tool in tools}


# LLM node
def call_llm(state: AgentState) -> AgentState:

    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages

    response = llm.invoke(messages)

    return {"messages": [response]}


# Tool execution node
def take_action(state: AgentState) -> AgentState:

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:

        print(f"\nCalling Tool: {t['name']}")

        if t['name'] not in tools_dict:

            result = "Tool does not exist."

        else:

            result = tools_dict[t['name']].invoke(
                t['args'].get("query", "")
            )

        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )

    return {"messages": results}


# Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)


graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "llm")

graph.set_entry_point("llm")


rag_agent = graph.compile()


def running_agent():

    print("\n===== RAG AGENT STARTED =====")

    while True:

        user_input = input("\nAsk a question: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n===== ANSWER =====")
        print(result["messages"][-1].content)


running_agent()
