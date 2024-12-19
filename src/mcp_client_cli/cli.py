#!/usr/bin/env python3

"""
Simple llm CLI that acts as MCP client.
"""

from datetime import datetime, timedelta
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Optional, List, Type, TypedDict
import uuid
import sys

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.tools import BaseTool, ToolException
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.managed import IsLastStep
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import BaseModel
from jsonschema_pydantic import jsonschema_to_pydantic
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

CACHE_EXPIRY_HOURS = 24
DEFAULT_QUERY = "Summarize https://www.youtube.com/watch?v=NExtKbS1Ljc"
CONFIG_FILE = 'mcp-server-config.json'
CONFIG_DIR = Path.home() / ".llm"
SQLITE_DB = CONFIG_DIR / "conversations.db"
CACHE_DIR = CONFIG_DIR / "mcp-tools"


def get_cached_tools(server_param: StdioServerParameters) -> Optional[List[types.Tool]]:
    """Retrieve cached tools if available and not expired.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
    
    Returns:
        Optional[List[types.Tool]]: A list of tools if cache is available and not expired, otherwise None.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{server_param.command}-{'-'.join(server_param.args)}".replace("/", "-")
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
        
    cache_data = json.loads(cache_file.read_text())
    cached_time = datetime.fromisoformat(cache_data["cached_at"])
    
    if datetime.now() - cached_time > timedelta(hours=CACHE_EXPIRY_HOURS):
        return None
            
    return [types.Tool(**tool) for tool in cache_data["tools"]]


def save_tools_cache(server_param: StdioServerParameters, tools: List[types.Tool]) -> None:
    """Save tools to cache.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
        tools (List[types.Tool]): The list of tools to be cached.
    """
    cache_key = f"{server_param.command}-{'-'.join(server_param.args)}".replace("/", "-")
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    cache_data = {
        "cached_at": datetime.now().isoformat(),
        "tools": [tool.model_dump() for tool in tools]
    }
    cache_file.write_text(json.dumps(cache_data))


def create_langchain_tool(
    tool_schema: types.Tool,
    server_params: StdioServerParameters
) -> BaseTool:
    """Create a LangChain tool from MCP tool schema.
    
    Args:
        tool_schema (types.Tool): The MCP tool schema.
        server_params (StdioServerParameters): The server parameters for the tool.
    
    Returns:
        BaseTool: The created LangChain tool.
    """
    input_model = jsonschema_to_pydantic(tool_schema.inputSchema)
    
    class McpTool(BaseTool):
        name: str = tool_schema.name
        description: str = tool_schema.description
        args_schema: Type[BaseModel] = input_model
        mcp_server_params: StdioServerParameters = server_params

        def _run(self, **kwargs):
            raise NotImplementedError("Only async operations are supported")

        async def _arun(self, **kwargs):
            async with stdio_client(self.mcp_server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, arguments=kwargs)
                    if result.isError:
                        raise ToolException(result.content)
                    return result.content
    
    return McpTool()


async def convert_mcp_to_langchain_tools(server_params: List[StdioServerParameters]) -> List[BaseTool]:
    """Convert MCP tools to LangChain tools.
    
    Args:
        server_params (List[StdioServerParameters]): A list of server parameters for MCP tools.
    
    Returns:
        List[BaseTool]: A list of converted LangChain tools.
    """
    langchain_tools = []
    
    for server_param in server_params:
        cached_tools = get_cached_tools(server_param)
        
        if cached_tools:
            for tool in cached_tools:
                langchain_tools.append(create_langchain_tool(tool, server_param))
            continue
            
        async with stdio_client(server_param) as (read, write):
            async with ClientSession(read, write) as session:
                print(f"Gathering capability of {server_param.command} {' '.join(server_param.args)}")
                await session.initialize()
                tools: types.ListToolsResult = await session.list_tools()
                save_tools_cache(server_param, tools.tools)
                
                for tool in tools.tools:
                    langchain_tools.append(create_langchain_tool(tool, server_param))
    
    return langchain_tools


# The AgentState class is used to maintain the state of the agent during a conversation.
class AgentState(TypedDict):
    # A list of messages exchanged in the conversation.
    messages: Annotated[list[BaseMessage], add_messages]
    # A flag indicating whether the current step is the last step in the conversation.
    is_last_step: IsLastStep
    # The current date and time, used for context in the conversation.
    today_datetime: str


class ConversationManager:
    """Manages conversation persistence in SQLite database."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def _init_db(self, db) -> None:
        """Initialize database schema.
        
        Args:
            db: The database connection object.
        """
        await db.execute("""
            CREATE TABLE IF NOT EXISTS last_conversation (
                id INTEGER PRIMARY KEY,
                thread_id TEXT NOT NULL
            )
        """)
        await db.commit()
    
    async def get_last_id(self) -> str:
        """Get the thread ID of the last conversation.
        
        Returns:
            str: The thread ID of the last conversation, or a new UUID if no conversation exists.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            async with db.execute("SELECT thread_id FROM last_conversation LIMIT 1") as cursor:
                row = await cursor.fetchone()
            return row[0] if row else uuid.uuid4().hex
    
    async def save_id(self, thread_id: str, db = None) -> None:
        """Save thread ID as the last conversation.
        
        Args:
            thread_id (str): The thread ID to save.
            db: The database connection object (optional).
        """
        if db is None:
            async with aiosqlite.connect(self.db_path) as db:
                await self._save_id(db, thread_id)
        else:
            await self._save_id(db, thread_id)
    
    async def _save_id(self, db, thread_id: str) -> None:
        """Internal method to save thread ID.
        
        Args:
            db: The database connection object.
            thread_id (str): The thread ID to save.
        """
        async with db.cursor() as cursor:
            await self._init_db(db)
            await cursor.execute("DELETE FROM last_conversation")
            await cursor.execute(
                "INSERT INTO last_conversation (thread_id) VALUES (?)", 
                (thread_id,)
            )
            await db.commit()


async def run() -> None:
    """Run the LangChain agent with MCP tools.
    
    This function initializes the agent, loads the configuration, and processes the query.
    """
    # Argument parsing and query determination
    parser = argparse.ArgumentParser(description='Run LangChain agent with MCP tools')
    parser.add_argument('query', nargs='*', default=[],
                       help='The query to process (default: read from stdin or use default query)')
    args = parser.parse_args()
    
    # Check if there's input from stdin (pipe)
    if not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        # Use command line args or default query
        query = ' '.join(args.query) if args.query else DEFAULT_QUERY
    
    if args.query and args.query[0] == "commit":
        query = "check git status and diff. Then commit it with descriptive and concise commit msg"

    # Configuration file loading
    config_paths = [CONFIG_FILE, CONFIG_DIR / "config.json"]
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                server_config = json.load(f)
            break
    else:
        raise FileNotFoundError(f"Could not find config file in any of: {', '.join(config_paths)}")
    
    # Server parameters initialization
    server_params = [
        StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env={**config.get("env", {}), **os.environ}
        )
        for config in server_config["mcpServers"].values()
    ]

    # LangChain tools conversion
    langchain_tools = await convert_mcp_to_langchain_tools(server_params)
    
    # Model initialization
    llm_config = server_config.get("llm", {})
    model = init_chat_model(
        model=llm_config.get("model", "gpt-4o"),
        model_provider=llm_config.get("provider", "openai"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0),
        base_url=llm_config.get("base_url")
    )
    
    # Prompt creation
    prompt = ChatPromptTemplate.from_messages([
        ("system", server_config["systemPrompt"]),
        ("placeholder", "{messages}")
    ])

    # Conversation manager initialization
    conversation_manager = ConversationManager(SQLITE_DB)
    
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB) as checkpointer:
        # Agent executor creation
        agent_executor = create_react_agent(
            model, 
            langchain_tools, 
            state_schema=AgentState, 
            state_modifier=prompt,
            checkpointer=checkpointer
        )
        
        # Query processing and continuation check
        is_continuation = query.startswith('c ')
        if is_continuation:
            query = query[2:]  # Remove 'c ' prefix
            thread_id = await conversation_manager.get_last_id()
        else:
            thread_id = uuid.uuid4().hex
        input_messages = {
            "messages": [HumanMessage(content=query)], 
            "today_datetime": datetime.now().isoformat(),
        }
        
        # Message streaming and tool calls handling
        async for chunk in agent_executor.astream(
            input_messages,
            stream_mode=["messages", "values"],
            config={"configurable": {"thread_id": thread_id}}
        ):
            # If this is a message chunk
            if isinstance(chunk, tuple) and chunk[0] == "messages":
                message_chunk = chunk[1][0]  # Get the message content
                if isinstance(message_chunk, AIMessageChunk):
                    content = message_chunk.content
                    if isinstance(content, str):
                        print(content, end="", flush=True)
                    elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                        print(content[0]["text"], end="", flush=True)
            # If this is a final value
            elif isinstance(chunk, dict) and "messages" in chunk:
                # Print a newline after the complete message
                print("\n", flush=True)
            elif isinstance(chunk, tuple) and chunk[0] == "values":
                message = chunk[1]['messages'][-1]
                if isinstance(message, AIMessage) and message.tool_calls:
                    print("\n\nTool Calls:")
                    for tc in message.tool_calls:
                        lines = [
                            f"  {tc.get('name', 'Tool')}",
                        ]
                        if tc.get("error"):
                            lines.append(f"  Error: {tc.get('error')}")
                        lines.append("  Args:")
                        args = tc.get("args")
                        if isinstance(args, str):
                            lines.append(f"    {args}")
                        elif isinstance(args, dict):
                            for arg, value in args.items():
                                lines.append(f"    {arg}: {value}")
                        print("\n".join(lines))
        print()

        # Saving the last conversation thread ID
        await conversation_manager.save_id(thread_id, checkpointer.conn)


def main() -> None:
    """Entry point of the script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
