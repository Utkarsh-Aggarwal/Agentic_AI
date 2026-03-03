""" This is a simple agent that uses a state graph to manage its conversation history. It loads the conversation history from a file, processes user input, generates a response using a language model, and saves the conversation history back to the file. The agent continues to interact with the user until they type "exit". """
from typing import TypedDict,List, Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

class AgentState(TypedDict):
  messages: List[Union[HumanMessage,AIMessage]]
  
def process(state: AgentState) -> AgentState:
  """ this node will solve the request you input """
  reponse = llm.invoke(state['messages'])
  state["messages"].append(AIMessage(content=reponse.content))
  print({reponse.content})
  return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process") 
graph.add_edge("process", END)
agent = graph.compile()

def load_history(file_path):
    history = []

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("User: "):
                history.append(line.replace("User: ", "").strip())
            elif line.startswith("AI: "):
                history.append(line.replace("AI: ", "").strip())

    except FileNotFoundError:
        pass  # first run, no file yet

    return history

def save_message(file_path, role, content):
    with open(file_path, "a") as f:
        f.write(f"{role}: {content}\n")

file_path = "logging.txt"


conversation_history = load_history(file_path)


user_input = input("Enter your message: ")

while user_input.lower() != "exit":
   conversation_history.append(HumanMessage(content=user_input))
   result = agent.invoke({"messages": conversation_history})
   print(result["messages"])
   conversation_history = result["messages"]
   ai_text = result["messages"][-1].content
   save_message(file_path, "AI", ai_text)
   user_input = input("Enter your message: ")

with open("logging.txt", "w") as file:
  for message in conversation_history:
    if isinstance(message, HumanMessage):
      file.write(f"User: {message.content}\n")
    elif isinstance(message, AIMessage):
      file.write(f"AI: {message.content}\n")
  file.write("End of conversation\n")

print("Conversation logged to logging.txt")