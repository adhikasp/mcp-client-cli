from typing import List, Type, Optional, Any, override
from pydantic import BaseModel
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from mcp import StdioServerParameters, types, ClientSession
from mcp.client.stdio import stdio_client
import pydantic
from pydantic_core import core_schema as cs
from pydantic.json_schema import JsonSchemaValue

from .storage import *

class McpServerConfig(BaseModel):
    """Configuration for an MCP server.
    
    This class represents the configuration needed to connect to and identify an MCP server,
    containing both the server's name and its connection parameters.

    Attributes:
        server_name (str): The name identifier for this MCP server
        server_param (StdioServerParameters): Connection parameters for the server, including
            command, arguments and environment variables
    """
    
    server_name: str
    server_param: StdioServerParameters

class McpToolkit(BaseToolkit):
    name: str
    server_param: StdioServerParameters
    _session: Optional[ClientSession] = None
    _tools: List[BaseTool] = []
    _client = None
    _prompts: List[types.Prompt] = []

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    async def _start_session(self):
        if self._session:
            return self._session

        self._client = stdio_client(self.server_param)
        read, write = await self._client.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self._session

    async def initialize(self, force_refresh: bool = False):
        cached_tools = get_cached_tools(self.server_param)
        cached_prompts = get_cached_prompts(self.server_param)
        if not force_refresh and (cached_tools or cached_prompts):
            for tool in cached_tools:
                self._tools.append(create_langchain_tool(tool, self._session, self))
            self._prompts = cached_prompts
            return

        try:
            await self._start_session()
            tools: types.ListToolsResult = await self._session.list_tools()
            try:
                prompts: types.ListPromptsResult = await self._session.list_prompts()
                self._prompts = prompts.prompts
            except Exception as e:
                pass
            save_tools_cache(self.server_param, tools.tools, self._prompts)
            for tool in tools.tools:
                self._tools.append(create_langchain_tool(tool, self._session, self))
        except Exception as e:
            print(f"Error gathering tools for {self.server_param.command} {' '.join(self.server_param.args)}: {e}")
            raise e
        
    async def close(self):
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
        except:
            # Currently above code doesn't really works and not closing the session
            # But it's not a big deal as we are exiting anyway
            # TODO find a way to cleanly close the session
            pass
        try:
            if self._client:
                await self._client.__aexit__(None, None, None)
        except:
            # TODO find a way to cleanly close the client
            pass

    def get_tools(self) -> List[BaseTool]:
        return self._tools

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
            if result.isError:
                raise ToolException(result.content)
            return result.content

        @override
        @property
        def tool_call_schema(self) -> type[pydantic.BaseModel]:
            assert self.args_schema is not None  # noqa: S101
            return self.args_schema
    
    try:
        return McpTool(
            name=tool_schema.name,
            description=tool_schema.description,
            # args_schema=jsonschema_to_pydantic(tool_schema.inputSchema),
            session=session,
            toolkit=toolkit,
            toolkit_name=toolkit.name,
            args_schema=create_schema_model(tool_schema.inputSchema),
        )
    except Exception as e:
        print(f"Error creating tool {tool_schema.name}: {e}")
        print("Tool schema: ", tool_schema.inputSchema)
        raise e

def create_schema_model(schema: dict[str, Any]) -> type[pydantic.BaseModel]:
    # Create a new model class that returns our JSON schema.
    # LangChain requires a BaseModel class.
    class Schema(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

        @override
        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: cs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return schema

    return Schema

async def convert_mcp_to_langchain_tools(server_config: McpServerConfig, force_refresh: bool = False) -> McpToolkit:
    """Convert MCP tools to LangChain tools and create a toolkit.
    
    Args:
        server_config (McpServerConfig): Configuration for the MCP server including name and parameters.
        force_refresh (bool, optional): Whether to force refresh the tools cache. Defaults to False.
    
    Returns:
        McpToolkit: A toolkit containing the converted LangChain tools.
    """
    toolkit = McpToolkit(name=server_config.server_name, server_param=server_config.server_param)
    await toolkit.initialize(force_refresh=force_refresh)
    return toolkit