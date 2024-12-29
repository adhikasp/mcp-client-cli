#!/usr/bin/env python3

"""
Simple llm CLI that acts as MCP client.
"""

from datetime import datetime
import argparse
import asyncio
import os
from typing import Annotated, TypedDict
import uuid
import sys
import re
import anyio
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.managed import IsLastStep
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.table import Table

from .const import *
from .output import *
from .storage import *
from .tool import *
from .prompt import *
from .memory import *
from .config import AppConfig

# The AgentState class is used to maintain the state of the agent during a conversation.
class AgentState(TypedDict):
    # A list of messages exchanged in the conversation.
    messages: Annotated[list[BaseMessage], add_messages]
    # A flag indicating whether the current step is the last step in the conversation.
    is_last_step: IsLastStep
    # The current date and time, used for context in the conversation.
    today_datetime: str
    # The user's memories.
    memories: str = "no memories"

async def run() -> None:
    """Run the LLM agent."""
    args = setup_argument_parser()
    query, is_conversation_continuation = parse_query(args)
    app_config = AppConfig.load()
    
    if args.list_tools:
        await handle_list_tools(app_config, args)
        return
    
    if args.show_memories:
        await handle_show_memories()
        return
        
    if args.list_prompts:
        handle_list_prompts()
        return
        
    await handle_conversation(args, query, is_conversation_continuation, app_config)

def setup_argument_parser() -> argparse.Namespace:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(
        description='Run LangChain agent with MCP tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm "What is the capital of France?"     Ask a simple question
  llm c "tell me more"                     Continue previous conversation
  llm p review                             Use a prompt template
  cat file.txt | llm                       Process input from a file
  llm --list-tools                         Show available tools
  llm --list-prompts                       Show available prompt templates
  llm --no-confirmations "search web"      Run tools without confirmation
        """
    )
    parser.add_argument('query', nargs='*', default=[],
                       help='The query to process (default: read from stdin). '
                            'Special prefixes:\n'
                            '  c: Continue previous conversation\n'
                            '  p: Use prompt template')
    parser.add_argument('--list-tools', action='store_true',
                       help='List all available LLM tools')
    parser.add_argument('--list-prompts', action='store_true',
                       help='List all available prompts')
    parser.add_argument('--no-confirmations', action='store_true',
                       help='Bypass tool confirmation requirements')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of tools capabilities')
    parser.add_argument('--text-only', action='store_true',
                       help='Print output as raw text instead of parsing markdown')
    parser.add_argument('--no-tools', action='store_true',
                       help='Do not add any tools')
    parser.add_argument('--show-memories', action='store_true',
                       help='Show user memories')
    return parser.parse_args()

async def handle_list_tools(app_config: AppConfig, args: argparse.Namespace) -> None:
    """Handle the --list-tools command."""
    server_configs = [
        McpServerConfig(
            server_name=name,
            server_param=StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env={**(config.env or {}), **os.environ}
            ),
            exclude_tools=config.exclude_tools or []
        )
        for name, config in app_config.get_enabled_servers().items()
    ]
    toolkits, tools = await load_tools(server_configs, args.no_tools, args.force_refresh)
    
    console = Console()
    table = Table(title="Available LLM Tools")
    table.add_column("Toolkit", style="cyan")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")

    for tool in tools:
        if isinstance(tool, McpTool):
            table.add_row(tool.toolkit_name, tool.name, tool.description)

    console.print(table)

    for toolkit in toolkits:
        await toolkit.close()

async def handle_show_memories() -> None:
    """Handle the --show-memories command."""
    store = SqliteStore(SQLITE_DB)
    memories = await get_memories(store)
    console = Console()
    table = Table(title="My LLM Memories")
    for memory in memories:
        table.add_row(memory)
    console.print(table)

def handle_list_prompts() -> None:
    """Handle the --list-prompts command."""
    console = Console()
    table = Table(title="Available Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Template")
    table.add_column("Arguments")
    
    for name, template in prompt_templates.items():
        table.add_row(name, template, ", ".join(re.findall(r'\{(\w+)\}', template)))
        
    console.print(table)

async def load_tools(server_configs: list[McpServerConfig], no_tools: bool, force_refresh: bool) -> tuple[list, list]:
    """Load and convert MCP tools to LangChain tools."""
    if no_tools:
        return [], []
        
    toolkits = []
    langchain_tools = []
    
    async def convert_toolkit(server_config: McpServerConfig):
        toolkit = await convert_mcp_to_langchain_tools(server_config, force_refresh)
        toolkits.append(toolkit)
        langchain_tools.extend(toolkit.get_tools())

    async with anyio.create_task_group() as tg:
        for server_param in server_configs:
            tg.start_soon(convert_toolkit, server_param)
            
    langchain_tools.append(save_memory)
    return toolkits, langchain_tools

async def handle_conversation(args: argparse.Namespace, query: str, 
                            is_conversation_continuation: bool, app_config: AppConfig) -> None:
    """Handle the main conversation flow."""
    server_configs = [
        McpServerConfig(
            server_name=name,
            server_param=StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env={**(config.env or {}), **os.environ}
            ),
            exclude_tools=config.exclude_tools or []
        )
        for name, config in app_config.get_enabled_servers().items()
    ]
    toolkits, tools = await load_tools(server_configs, args.no_tools, args.force_refresh)
    
    model = init_chat_model(
        model=app_config.llm.model,
        model_provider=app_config.llm.provider,
        api_key=app_config.llm.api_key,
        temperature=app_config.llm.temperature,
        base_url=app_config.llm.base_url,
        default_headers={
            "X-Title": "mcp-client-cli",
            "HTTP-Referer": "https://github.com/adhikasp/mcp-client-cli",
        },
        extra_body={"transforms": ["middle-out"]}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.system_prompt),
        ("placeholder", "{messages}")
    ])

    conversation_manager = ConversationManager(SQLITE_DB)
    
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB) as checkpointer:
        store = SqliteStore(SQLITE_DB)
        memories = await get_memories(store)
        agent_executor = create_react_agent(
            model, tools, state_schema=AgentState, 
            state_modifier=prompt, checkpointer=checkpointer, store=store
        )
        
        thread_id = (await conversation_manager.get_last_id() if is_conversation_continuation 
                    else uuid.uuid4().hex)

        input_messages = AgentState(
            messages=[HumanMessage(content=query)], 
            today_datetime=datetime.now().isoformat(),
            memories=memories,
        )

        output = OutputHandler(text_only=args.text_only)
        output.start()
        try:
            async for chunk in agent_executor.astream(
                input_messages,
                stream_mode=["messages", "values"],
                config={"configurable": {"thread_id": thread_id, "user_id": "myself"}, 
                       "recursion_limit": 100}
            ):
                output.update(chunk)
                if not args.no_confirmations:
                    if not output.confirm_tool_call(app_config.__dict__, chunk):
                        break
        except Exception as e:
            output.update_error(e)
        finally:
            output.finish()

        await conversation_manager.save_id(thread_id, checkpointer.conn)

    for toolkit in toolkits:
        await toolkit.close()

def parse_query(args: argparse.Namespace) -> tuple[str, bool]:
    """
    Parse the query from command line arguments.
    Returns a tuple of (query, is_conversation_continuation).
    """
    query_parts = ' '.join(args.query).split()

    # No arguments provided
    if not query_parts:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip(), False
        return '', False

    # Check for conversation continuation
    if query_parts[0] == 'c':
        return ' '.join(query_parts[1:]), True

    # Check for prompt template
    if query_parts[0] == 'p' and len(query_parts) >= 2:
        template_name = query_parts[1]
        if template_name not in prompt_templates:
            print(f"Error: Prompt template '{template_name}' not found.")
            print("Available templates:", ", ".join(prompt_templates.keys()))
            return '', False

        template = prompt_templates[template_name]
        template_args = query_parts[2:]
        
        try:
            # Extract variable names from the template
            var_names = re.findall(r'\{(\w+)\}', template)
            # Create dict mapping parameter names to arguments
            template_vars = dict(zip(var_names, template_args))
            return template.format(**template_vars), False
        except KeyError as e:
            print(f"Error: Missing argument {e}")
            return '', False

    # Regular query
    return ' '.join(query_parts), False

def main() -> None:
    """Entry point of the script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
