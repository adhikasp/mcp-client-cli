from datetime import datetime, timedelta
from typing import Optional, List
from mcp import StdioServerParameters, types
import json
import aiosqlite
import uuid

from .const import *

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


def get_cached_prompts(server_param: StdioServerParameters) -> Optional[List[types.Prompt]]:
    """Retrieve cached prompts if available and not expired.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
    
    Returns:
        Optional[List[types.Prompt]]: A list of prompts if cache is available and not expired, otherwise None.
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
            
    return [types.Prompt(**prompt) for prompt in cache_data["prompts"]] if "prompts" in cache_data else None


def save_tools_cache(server_param: StdioServerParameters, tools: List[types.Tool], prompts: Optional[List[types.Prompt]] = None) -> None:
    """Save tools and prompts to cache.
    
    Args:
        server_param (StdioServerParameters): The server parameters to identify the cache.
        tools (List[types.Tool]): The list of tools to be cached.
        prompts (Optional[List[types.Prompt]]): The list of prompts to be cached.
    """
    cache_key = f"{server_param.command}-{'-'.join(server_param.args)}".replace("/", "-")
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    cache_data = {
        "cached_at": datetime.now().isoformat(),
        "tools": [tool.model_dump() for tool in tools],
    }
    if prompts:
        cache_data["prompts"] = [prompt.model_dump() for prompt in prompts]
    cache_file.write_text(json.dumps(cache_data))


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