"""
Simple agent that takes user input and generates a response using a language model.
"""


from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

class AgentState(TypedDict):
  messages: List[HumanMessage]

def process(state: AgentState) -> AgentState:
  # Process the state and generate a response
  response = llm.invoke(state['messages'])
  print({response.content})
  return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process") 
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})