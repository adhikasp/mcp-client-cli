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
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from langgraph.managed import IsLastStep
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.table import Table
import base64
import imghdr as imghdr
import mimetypes
from dotenv import load_dotenv, find_dotenv

from .input import *
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
    remaining_steps: int = 5

async def run() -> None:
    """Run the LLM agent."""
    load_dotenv(find_dotenv())
    
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
    parser.add_argument('--no-intermediates', action='store_true',
                       help='Only print the final message')
    parser.add_argument('--show-memories', action='store_true',
                       help='Show user memories')
    parser.add_argument('--model',
                       help='Override the model specified in config')
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

async def handle_conversation(args: argparse.Namespace, query: HumanMessage, 
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
    
    extra_body = {}
    if app_config.llm.base_url and "openrouter" in app_config.llm.base_url:
        extra_body = {"transforms": ["middle-out"]}
    # Override model if specified in command line
    if args.model:
        app_config.llm.model = args.model
    
    if app_config.llm.provider == "gigachat":
        from langchain_gigachat import GigaChat
        
        model: BaseChatModel = GigaChat(
            model=app_config.llm.model,
            model_provider=app_config.llm.provider,
            api_key=app_config.llm.api_key,
            temperature=app_config.llm.temperature,
            base_url=app_config.llm.base_url,
            default_headers={
                "X-Title": "mcp-client-cli",
                "HTTP-Referer": "https://github.com/adhikasp/mcp-client-cli",
            },
            extra_body=extra_body
        )
    else:
        model: BaseChatModel = init_chat_model(
            model=app_config.llm.model,
            model_provider=app_config.llm.provider,
            api_key=app_config.llm.api_key,
            temperature=app_config.llm.temperature,
            base_url=app_config.llm.base_url,
            default_headers={
                "X-Title": "mcp-client-cli",
                "HTTP-Referer": "https://github.com/adhikasp/mcp-client-cli",
            },
            extra_body=extra_body
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", app_config.system_prompt),
        ("placeholder", "{messages}")
    ])

    conversation_manager = ConversationManager(SQLITE_DB)
    
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB) as checkpointer:
        store = SqliteStore(SQLITE_DB)
        memories = await get_memories(store)
        formatted_memories = "\n".join(f"- {memory}" for memory in memories)
        agent_executor = create_react_agent(
            model, tools, state_schema=AgentState, 
            state_modifier=prompt, checkpointer=checkpointer, store=store
        )
        
        thread_id = (await conversation_manager.get_last_id() if is_conversation_continuation 
                    else uuid.uuid4().hex)

        input_messages = AgentState(
            messages=[query], 
            today_datetime=datetime.now().isoformat(),
            memories=formatted_memories,
            remaining_steps=3
        )

        output = OutputHandler(text_only=args.text_only, only_last_message=args.no_intermediates)
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

def parse_query(args: argparse.Namespace) -> tuple[HumanMessage, bool]:
    """
    Parse the query from command line arguments.
    Returns a tuple of (HumanMessage, is_conversation_continuation).
    """
    query_parts = ' '.join(args.query).split()
    stdin_content = ""
    stdin_image = None
    is_continuation = False

    # Handle clipboard content if requested
    if query_parts and query_parts[0] == 'cb':
        # Remove 'cb' from query parts
        query_parts = query_parts[1:]
        # Try to get content from clipboard
        clipboard_result = get_clipboard_content()
        if clipboard_result:
            content, mime_type = clipboard_result
            if mime_type:  # It's an image
                stdin_image = base64.b64encode(content).decode('utf-8')
            else:  # It's text
                stdin_content = content
        else:
            print("No content found in clipboard")
            raise Exception("Clipboard is empty")
    # Check if there's input from pipe
    elif not sys.stdin.isatty():
        stdin_data = sys.stdin.buffer.read()
        # Try to detect if it's an image
        image_type = imghdr.what(None, h=stdin_data)
        if image_type:
            # It's an image, encode it as base64
            stdin_image = base64.b64encode(stdin_data).decode('utf-8')
            mime_type = mimetypes.guess_type(f"dummy.{image_type}")[0] or f"image/{image_type}"
        else:
            # It's text
            stdin_content = stdin_data.decode('utf-8').strip()

    # Process the query text
    query_text = ""
    if query_parts:
        if query_parts[0] == 'c':
            is_continuation = True
            query_text = ' '.join(query_parts[1:])
        elif query_parts[0] == 'p' and len(query_parts) >= 2:
            template_name = query_parts[1]
            if template_name not in prompt_templates:
                print(f"Error: Prompt template '{template_name}' not found.")
                print("Available templates:", ", ".join(prompt_templates.keys()))
                return HumanMessage(content=""), False

            template = prompt_templates[template_name]
            template_args = query_parts[2:]
            try:
                # Extract variable names from the template
                var_names = re.findall(r'\{(\w+)\}', template)
                # Create dict mapping parameter names to arguments
                template_vars = dict(zip(var_names, template_args))
                query_text = template.format(**template_vars)
            except KeyError as e:
                print(f"Error: Missing argument {e}")
                return HumanMessage(content=""), False
        else:
            query_text = ' '.join(query_parts)

    # Combine stdin content with query text if both exist
    if stdin_content and query_text:
        query_text = f"{stdin_content}\n\n{query_text}"
    elif stdin_content:
        query_text = stdin_content
    elif not query_text and not stdin_image:
        return HumanMessage(content=""), False

    # Create the message content
    if stdin_image:
        content = [
            {"type": "text", "text": query_text or "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{stdin_image}"}}
        ]
    else:
        content = query_text

    return HumanMessage(content=content), is_continuation

def main() -> None:
    """Entry point of the script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
