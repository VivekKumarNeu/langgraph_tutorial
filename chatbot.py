from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


llm = init_chat_model(
    "llama3.2",
    model_provider="ollama",
    temperature=0
)

@tool
def get_stock_price(symbol: str) -> float:
    """Returns the stock price of a given symbol."""
    prices = {"MSFT": 200.3, "AMZN": 87.1}
    return prices.get(symbol.upper(), 0.0)

@tool
def buy_stocks(symbol: str, quantity: int) -> str:
    """Buy stocks of a given symbol and quantity."""
    return f"You bought {symbol}: {quantity}"

tools = [get_stock_price, buy_stocks]
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    # This keeps a running history of the conversation
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    # It's good practice to ensure there's a system message if the model struggles
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

memory = MemorySaver()

# 2. Build Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)

config = {'configurable': { 'thread_id' : '1'}}
# 3. Run Graph
# Wrap the message in a list []
input_data = {"messages": [{"role": "user", "content": "who w224as the first person to land on moon?"}]}

# Use a try-except or just print the full state to see what's happening
response = graph.invoke(input_data,config=config)
print("Final Response:", response["messages"][-1].content)

# 3. Run Graph
# Wrap the message in a list []
input_data = {"messages": [{"role": "user", "content": "I want to buy 20 AMZN stocks using current price. Then 15 MSFT. What will be the total cost?"}]}
response = graph.invoke(input_data, config=config)

# Print the last message in the conversation (the final answer)
print("Final Response:", response["messages"][-1].content)

input_data = {"messages": [{"role": "user", "content": "using the current price tell me the total price of 10 MSFT stocks and add it to the previous total "}]}
response = graph.invoke(input_data, config=config)

# Print the last message in the conversation (the final answer)
print("Final Response:", response["messages"][-1].content)


input_data = {"messages": [{"role": "user", "content": "Buy 10 MSFT stocks "}]}
response = graph.invoke(input_data, config=config)

# Print the last message in the conversation (the final answer)
print("Final Response:", response["messages"][-1].content)