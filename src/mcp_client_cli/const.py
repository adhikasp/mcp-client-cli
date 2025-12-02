import httpx
from datetime import timedelta
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

CACHE_EXPIRY_HOURS = 24
DEFAULT_QUERY = "Summarize https://www.youtube.com/watch?v=NExtKbS1Ljc"
CONFIG_FILE = 'mcp-server-config.json'
CONFIG_DIR = Path.home() / ".llm"
SQLITE_DB = CONFIG_DIR / "conversations.db"
CACHE_DIR = CONFIG_DIR / "mcp-tools"


class McpType(Enum):
    STDIO = 1
    SSE = 2
    STREAMABLE_HTTP = 3


class StramableHttpOrSseParameters(BaseModel):
    url: str
    headers: dict[str, str] | None = None
    timeout: float = 30
    sse_read_timeout: float = 60 * 5
    terminate_on_close: bool = True
