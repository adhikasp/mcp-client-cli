"""Configuration management for the MCP client CLI."""

from dataclasses import dataclass
from pathlib import Path
import os
import commentjson
from typing import Dict, List, Optional

from .const import CONFIG_FILE, CONFIG_DIR

@dataclass
class LLMConfig:
    """Configuration for the LLM model."""
    model: str = "gpt-4o"
    provider: str = "openai"
    api_key: Optional[str] = None
    temperature: float = 0
    base_url: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "LLMConfig":
        """Create LLMConfig from dictionary."""
        return cls(
            model=config.get("model", cls.model),
            provider=config.get("provider", cls.provider),
            api_key=config.get("api_key", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))),
            temperature=config.get("temperature", cls.temperature),
            base_url=config.get("base_url"),
        )

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    enabled: bool = True
    exclude_tools: List[str] = None
    requires_confirmation: List[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "ServerConfig":
        """Create ServerConfig from dictionary."""
        return cls(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env", {}),
            enabled=config.get("enabled", True),
            exclude_tools=config.get("exclude_tools", []),
            requires_confirmation=config.get("requires_confirmation", [])
        )

@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig
    system_prompt: str
    mcp_servers: Dict[str, ServerConfig]
    tools_requires_confirmation: List[str]

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from file."""
        config_paths = [CONFIG_FILE, CONFIG_DIR / "config.json"]
        chosen_path = next((path for path in config_paths if os.path.exists(path)), None)
        
        if chosen_path is None:
            raise FileNotFoundError(f"Could not find config file in any of: {', '.join(map(str, config_paths))}")

        with open(chosen_path, 'r') as f:
            config = commentjson.load(f)

        missing_keys = [key for key in ("systemPrompt", "mcpServers") if key not in config]
        if missing_keys:
            raise ValueError(
                f"Invalid config file {chosen_path}: "
                f"missing required key(s): {', '.join(missing_keys)}"
            )

        # Extract tools requiring confirmation
        tools_requires_confirmation = []
        for server_config in config["mcpServers"].values():
            tools_requires_confirmation.extend(server_config.get("requires_confirmation", []))

        return cls(
            llm=LLMConfig.from_dict(config.get("llm", {})),
            system_prompt=config["systemPrompt"],
            mcp_servers={
                name: ServerConfig.from_dict(server_config)
                for name, server_config in config["mcpServers"].items()
            },
            tools_requires_confirmation=tools_requires_confirmation
        )

    def get_enabled_servers(self) -> Dict[str, ServerConfig]:
        """Get only enabled server configurations."""
        return {
            name: config 
            for name, config in self.mcp_servers.items() 
            if config.enabled
        } 