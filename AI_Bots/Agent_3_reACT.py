from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #
from langchain_core.messages import ToolMessage #
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides additinal context without affecting the type
# Sequence - to automatically handle the state updates for sequences such as by adding new messages to the list of messages in the state


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int) -> int:
    #docstring is necessary 
    """Adds two numbers together."""
    return a + b

def subtract(a:int, b:int) -> int:
    #docstring is necessary 
    """Subtracts two numbers."""
    return a - b

def multiply(a:int, b:int) -> int:
    #docstring is necessary 
    """Multiplies two numbers together."""
    return a * b

tools = [add, subtract, multiply]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="you are my ai assistant , please answer my query in minimum words possible")
    response =  llm.invoke([system_prompt] + state['messages'])
    print({response.content})
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue, {"continue": "tools", "end": END})

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "what is 2 + 3 ?,what is 6 - 3 ? and what is 4 * 5 ? also tell me a joke ?")]}
print_stream(app.stream(inputs,stream_mode="values"))