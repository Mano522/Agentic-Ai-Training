
from typing import Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from tools import TOOLS


llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


class AgentState(MessagesState):
    route: str


# --------------------------------------------------
# 1. ROUTER NODE
# --------------------------------------------------

def router_node(state: AgentState):

    question = state["messages"][-1].content

    prompt = f"""
    Classify the following user question into exactly
    one of these categories:

    CALCULATION:
    Numerical calculations or mathematical operations.

    DATASET:
    Searching or retrieving information from the local CSV dataset.

    DOCUMENT:
    Questions about uploaded research PDFs, research papers,
    their methods, results, findings, comparisons or conclusions.

    GENERAL:
    General knowledge, explanations, comparisons or other questions
    that do not specifically require the uploaded documents or CSV.

    User question: {question}

    Return only one word:
    CALCULATION, DATASET, DOCUMENT or GENERAL.
    """

    response = llm.invoke(prompt)

    category = response.content.strip().upper()

    if "CALCULATION" in category:
        route = "calculation"

    elif "DATASET" in category:
        route = "dataset"

    elif "DOCUMENT" in category:
        route = "document"

    else:
        route = "general"

    return {"route": route}


# --------------------------------------------------
# 2. GENERAL RESPONSE NODE
# --------------------------------------------------

def general_node(state: AgentState):

    system_message = SystemMessage(
        content=(
            "You are a helpful research assistant. "
            "Provide clear, detailed and accurate explanations."
        )
    )

    response = llm.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}


# --------------------------------------------------
# 3. TOOL-BASED AGENT NODE
# --------------------------------------------------

def tool_agent_node(state: AgentState):

    tool_llm = llm.bind_tools(TOOLS)

    system_message = SystemMessage(
        content="""
        You are an Agentic Research Assistant.

        Use the available tools to complete the user's task.

        Tool selection:
        - Use calculator for numerical calculations.
        - Use dataset tools for information from the local CSV.
        - Use search_research_documents for questions about
          uploaded research PDFs.

        You may call multiple tools when required.

        For PDF-based answers:
        - Base document-specific claims on retrieved passages.
        - Include the source filename and page number.
        - Do not invent research findings or references.
        - If the retrieved passages do not contain the answer,
          clearly state that the available documents do not
          provide sufficient information.

        For other questions, provide clear and accurate answers.
        """
    )

    response = tool_llm.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}


# --------------------------------------------------
# 4. ROUTING DECISION
# --------------------------------------------------

def route_decision(state: AgentState) -> Literal[
    "calculation", "dataset", "document", "general"
]:

    return state["route"]


# --------------------------------------------------
# 5. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("general", general_node)
builder.add_node("tool_agent", tool_agent_node)
builder.add_node("tools", ToolNode(TOOLS))


builder.add_edge(START, "router")


builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "calculation": "tool_agent",
        "dataset": "tool_agent",
        "document": "tool_agent",
        "general": "general"
    }
)


builder.add_conditional_edges(
    "tool_agent",
    lambda state: (
        "tools" if state["messages"][-1].tool_calls else END
    ),
    {
        "tools": "tools",
        END: END
    }
)


builder.add_edge("tools", "tool_agent")
builder.add_edge("general", END)


# --------------------------------------------------
# 6. MEMORY AND COMPILATION
# --------------------------------------------------

memory = InMemorySaver()

agent = builder.compile(
    checkpointer=memory
)
