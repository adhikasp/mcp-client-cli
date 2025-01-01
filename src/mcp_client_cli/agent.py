from typing import Annotated, TypedDict, Sequence
import json

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore


# The AgentState class is used to maintain the state of the agent during a conversation.
class AgentState(TypedDict):
    # A list of messages exchanged in the conversation.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # A flag indicating whether the current step is the last step in the conversation.
    is_last_step: IsLastStep
    # The current date and time, used for context in the conversation.
    today_datetime: str
    # The user's memories.
    memories: str = "no memories"


class ReActAgent:
    """
    An agent that uses the ReAct architecture to respond to user queries.
    
    This mainly taken and modified from https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch
    """

    def __init__(self, model: BaseChatModel, tools: list[Tool], system_prompt: str, checkpointer: BaseCheckpointSaver, store: BaseStore):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])
        if tools:
            self.model = model.bind_tools(tools)
            self.tools_by_name = {tool.name: tool for tool in tools}
        else:
            self.model = model
            self.tools_by_name = {}
        self.chain = self.prompt | self.model
        self.tools = tools
        self.checkpointer = checkpointer
        self.store = store
        self.create_graph()

    async def astream(self, input: AgentState, thread_id: str):
        async for chunk in self.graph.astream(
                input, 
                stream_mode=["messages", "values"], 
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100},
            ):
            yield chunk

    def create_graph(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("tools", "agent")

        # Now we can compile and visualize our graph
        self.graph = workflow.compile(checkpointer=self.checkpointer, store=self.store)

    async def tool_node(self, state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = await self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    async def call_model(
        self,
        state: AgentState,
        config: RunnableConfig,
    ):
        response = await self.chain.ainvoke(state, config)
        return {"messages": [response]}


    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
