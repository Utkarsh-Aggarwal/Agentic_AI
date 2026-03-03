def process(state: AgentState) -> AgentState:
  """ this node will solve the request you input """
  reponse = llm.invoke(state['messages'])
  state["messages"].append(AIMessage(content=reponse.content))
  print({reponse.content})
  return state