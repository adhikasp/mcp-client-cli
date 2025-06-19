# MCP Client CLI Configuration

This document describes the configuration format for the MCP Client CLI. The configuration file uses JSON format (with support for comments via the `commentjson` library).

## Configuration File Location

The configuration file can be placed in either:
- `~/.llm/config.json` (user's home directory)
- `mcp-server-config.json` (in the current working directory)

## Configuration Structure

```json
{
  "systemPrompt": "string",
  "llm": {
    "provider": "string",
    "model": "string",
    "api_key": "string",
    "temperature": float,
    "base_url": "string"
  },
  "mcpServers": {
    "stdio_server_name": {
      "command": "string",
      "args": ["string"],
      "env": {
        "ENV_VAR_NAME": "value"
      },
      "enabled": boolean,
      "exclude_tools": ["string"],
      "requires_confirmation": ["string"]
    },
    "remote_server_name": {
      "url": "string",
      "headers": {},
      "timeout": float,
      "sse_read_timeout": float,
      "enabled": boolean,
      "exclude_tools": ["string"],
      "requires_confirmation": ["string"]
    }
  }
}
```

## Field Specifications

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `systemPrompt` | string | Yes | System prompt for the LLM |
| `llm` | object | No | LLM configuration |
| `mcpServers` | object | Yes | Dictionary of MCP server configurations |

### LLM Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `provider` | string | No | `"openai"` | LLM provider |
| `model` | string | No | `"gpt-4o"` | LLM model name |
| `api_key` | string | No | Environment vars | API key for the LLM service |
| `temperature` | float | No | `0` | Temperature for LLM responses |
| `base_url` | string | No | `null` | Custom API endpoint URL |

**Notes:**
- The `api_key` can be omitted if it's set via environment variables `LLM_API_KEY` or `OPENAI_API_KEY`

### MCP Server Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `command` | string | Yes (stdio mode) | - | Command to run the server |
| `args` | array | No | `[]` | Command-line arguments |
| `env` | object | No | `{}` | Environment variables |
| `url` | string | Yes (sse or streamable-http mode) | - | Sse or Streamable-HTTP url |
| `headers` | object | No | `{}` | Http request headers |
| `timeout` | number | No | 30 | Http request timeout |
| `sse_read_timeout` | number | No | 300 | Sse data read timeout |
| `enabled` | boolean | No | `true` | Whether the server is enabled |
| `exclude_tools` | array | No | `[]` | Tool names to exclude |
| `requires_confirmation` | array | No | `[]` | Tools requiring user confirmation |

## Example Configuration

```json
{
  "systemPrompt": "You are an AI assistant helping a software engineer...",
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "your-api-key-here",
    "temperature": 0
  },
  "mcpServers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "add": {
      "url": "http://localhost:8000/sse",
      "headers": {},
      "timeout": 50,
      "requires_confirmation": ["add"]
    },
    "subtract": {
      "url": "http://localhost:8000/mcp", 
      "headers": {},
      "timeout": 50
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key-here"
      }
    },
    "mcp-server-commands": {
      "command": "npx",
      "args": ["mcp-server-commands"],
      "requires_confirmation": [
        "run_command",
        "run_script"
      ]
    }
  }
}
```

## Comments in Configuration

The configuration file supports comments with `//` syntax:

```json
{
  "systemPrompt": "You are an AI assistant helping a software engineer...",
  // Uncomment this section to use Anthropic Claude
  // "llm": {
  //   "provider": "anthropic",
  //   "model": "claude-3-opus-20240229",
  //   "api_key": "your-anthropic-api-key"
  // },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "your-openai-api-key"
  }
}
```