
from langchain_core.messages import HumanMessage, ToolMessage
from agent import agent

print("=" * 60)
print("             AGENTIC RESEARCH ASSISTANT")
print("=" * 60)
print("Type 'exit' to stop.\n")

config = {
    "configurable": {
        "thread_id": "research-session-1"
    }
}

previous_message_count = 0

while True:

    question = input("\nYou: ")

    if question.lower().strip() in ["exit", "quit"]:
        print("\nResearch session ended.")
        break

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )

        new_messages = result["messages"][previous_message_count:]
        previous_message_count = len(result["messages"])

        print("\nExecution trace:")
        print("-" * 40)
        print("Selected route:", result.get("route", "unknown"))

        for message in new_messages:

            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    print("Tool:", call["name"])
                    print("Arguments:", call["args"])

            if isinstance(message, ToolMessage):
                print("Tool result:", message.content)

        print("-" * 40)
        print("\nAssistant:")
        print(result["messages"][-1].content)

    except Exception as e:
        print("\nExecution failed:", str(e))
        print("Please try rephrasing your question.")

