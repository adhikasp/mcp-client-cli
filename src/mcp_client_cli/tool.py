from typing import List, Type, Optional, Union
from pydantic import BaseModel
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from mcp import StdioServerParameters, types, ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
import pydantic
from pydantic_core import to_json
from jsonschema_pydantic import jsonschema_to_pydantic
import asyncio
from .storage import *
from .const import McpType, StramableHttpOrSseParameters

class McpServerConfig(BaseModel):
    """Configuration for an MCP server.
    
    This class represents the configuration needed to connect to and identify an MCP server,
    containing both the server's name and its connection parameters.

    Attributes:
        server_name (str): The name identifier for this MCP server
        server_param (StdioServerParameters): Connection parameters for the server, including
            command, arguments and environment variables
        exclude_tools (list[str]): List of tool names to exclude from this server
    """
    mcp_type: McpType
    server_name: str
    server_param: Union[StdioServerParameters, StramableHttpOrSseParameters]
    exclude_tools: list[str] = []


class McpToolkit(BaseToolkit):
    name: str
    mcp_type: McpType
    server_param: Union[StdioServerParameters, StramableHttpOrSseParameters]
    exclude_tools: list[str] = []
    _session: Optional[ClientSession] = None
    _tools: List[BaseTool] = []
    _client = None
    _init_lock: asyncio.Lock | None = None

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self._init_lock = asyncio.Lock()

    async def _start_session(self):
        async with self._init_lock:
            if self._session:
                return self._session
            read, write, get_session_id = None, None, None
            if self.mcp_type == McpType.STDIO:
                self._client = stdio_client(self.server_param)
                read, write = await self._client.__aenter__()
            elif self.mcp_type == McpType.STREAMABLE_HTTP:
                self._client = streamablehttp_client(url=self.server_param.url,
                                                     headers=self.server_param.headers,
                                                     timeout=self.server_param.timeout,
                                                     sse_read_timeout=self.server_param.sse_read_timeout,
                                                     terminate_on_close=self.server_param.terminate_on_close)
                read, write, _ = await self._client.__aenter__()
            elif self.mcp_type == McpType.SSE:
                self._client = sse_client(url=self.server_param.url,
                                          headers=self.server_param.headers,
                                          timeout=self.server_param.timeout,
                                          sse_read_timeout=self.server_param.sse_read_timeout,)
                read, write = await self._client.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            return self._session

    async def initialize(self, force_refresh: bool = False):
        if self._tools and not force_refresh:
            return

        cached_tools = get_cached_tools(self.server_param)
        if cached_tools and not force_refresh:
            for tool in cached_tools:
                if tool.name in self.exclude_tools:
                    continue
                self._tools.append(create_langchain_tool(tool, self._session, self))
            return

        try:
            await self._start_session()
            tools: types.ListToolsResult = await self._session.list_tools()
            save_tools_cache(self.server_param, tools.tools)
            for tool in tools.tools:
                if tool.name in self.exclude_tools:
                    continue
                self._tools.append(create_langchain_tool(tool, self._session, self))
        except Exception as e:
            print(f"Error gathering tools for {self.server_param.model_dump()}")
            raise e
        
    async def close(self):
        try:
            if self._session:
                try:
                    # Add timeout to prevent hanging
                    async with asyncio.timeout(2.0):
                        # Create a new task to handle cleanup in the correct context
                        await asyncio.create_task(self._session.__aexit__(None, None, None))
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    pass
        finally:
            try:
                if self._client:
                    try:
                        # Add timeout to prevent hanging
                        async with asyncio.timeout(2.0):
                            # Create a new task to handle cleanup in the correct context
                            await asyncio.create_task(self._client.__aexit__(None, None, None))
                    except asyncio.TimeoutError:
                        pass
                    except Exception as e:
                        pass
            except:
                pass

    def get_tools(self) -> List[BaseTool]:
        return self._tools


class McpTool(BaseTool):
    toolkit_name: str
    name: str
    description: str
    args_schema: Type[BaseModel]
    session: Optional[ClientSession]
    toolkit: McpToolkit

    handle_tool_error: bool = True

    def _run(self, **kwargs):
        raise NotImplementedError("Only async operations are supported")

    async def _arun(self, **kwargs):
        if not self.session:
            self.session = await self.toolkit._start_session()

        result = await self.session.call_tool(self.name, arguments=kwargs)
        content = to_json(result.content).decode()
        if result.isError:
            raise ToolException(content)
        return content

def create_langchain_tool(
    tool_schema: types.Tool,
    session: ClientSession,
    toolkit: McpToolkit,
) -> BaseTool:
    """Create a LangChain tool from MCP tool schema.
    
    Args:
        tool_schema (types.Tool): The MCP tool schema.
        session (ClientSession): The session for the tool.
    
    Returns:
        BaseTool: The created LangChain tool.
    """
    return McpTool(
        name=tool_schema.name,
        description=tool_schema.description,
        args_schema=jsonschema_to_pydantic(tool_schema.inputSchema),
        session=session,
        toolkit=toolkit,
        toolkit_name=toolkit.name,
    )


async def convert_mcp_to_langchain_tools(server_config: McpServerConfig, force_refresh: bool = False) -> McpToolkit:
    """Convert MCP tools to LangChain tools and create a toolkit.
    
    Args:
        server_config (McpServerConfig): Configuration for the MCP server including name and parameters.
        force_refresh (bool, optional): Whether to force refresh the tools cache. Defaults to False.
    
    Returns:
        McpToolkit: A toolkit containing the converted LangChain tools.
    """
    toolkit = McpToolkit(
        mcp_type=server_config.mcp_type,
        name=server_config.server_name, 
        server_param=server_config.server_param,
        exclude_tools=server_config.exclude_tools
    )
    await toolkit.initialize(force_refresh=force_refresh)
    return toolkit
