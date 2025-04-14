from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


memory = MemorySaver()


graph_builder = StateGraph(State)


model_name = "google_vertexai/gemini-2.0-flash-001"
model_name = "openai/gpt-4o"

# model = "gemini-2.0-flash-001"
model = "gpt-4o"

# llm = init_chat_model(
#     model, model_provider="google_vertexai", temperature=0.0, streaming=True
# )

llm = init_chat_model(model, model_provider="openai", temperature=0.0, streaming=True)


@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b


@tool
def concat_strings(a: str, b: str) -> str:
    """Concatenate two strings"""
    return str(a) + str(b)


tool_node = ToolNode(tools=[add_numbers, concat_strings])
llm_with_tools = llm.bind_tools(tools=[add_numbers, concat_strings])

# react_agent = create_react_agent(llm, tools=[add_numbers])


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Build nodes and graph
graph_builder.add_node(chatbot)

graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)

# Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}

# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )

        for event in events:
            event["messages"][-1].pretty_print()

    except Exception:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)

        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )

        for event in events:
            event["messages"][-1].pretty_print()
        break
