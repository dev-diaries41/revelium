from typing import List, Optional
import aiosqlite
import asyncio
from revelium.prompts.types import Prompt
from revelium.prompts.store.prompt_store import PromptStore

class SQLitePromptStore(PromptStore):
    """Async SQLite-backed PromptStore using aiosqlite with safe concurrent writes."""

    def __init__(self, db_path: str = "prompts.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()  # Serialize writes (SQLite allows multiple readers, one writer)
        self._init_done = False

    async def _init_db(self):
        """Initialize table if not exists."""
        if self._init_done:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL
                )
            """)
            await db.commit()
        self._init_done = True

    async def get(self, limit: Optional[int] = None, offset: int = 0) -> List[Prompt]:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT prompt_id, content FROM prompts ORDER BY prompt_id"
            params = ()
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params = (limit, offset)
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return [Prompt(prompt_id=row[0], content=row[1]) for row in rows]

    async def get_by_ids(self, ids: List[str]) -> List[Prompt]:
        if not ids:
            return []
        await self._init_db()
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT prompt_id, content FROM prompts WHERE prompt_id IN ({placeholders})"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, ids)
            rows = await cursor.fetchall()
            await cursor.close()
            return [Prompt(prompt_id=row[0], content=row[1]) for row in rows]

    async def add(self, items: List[Prompt]) -> None:
        if not items:
            return
        await self._init_db()
        async with self._lock:  # serialize writes
            async with aiosqlite.connect(self.db_path) as db:
                # Use INSERT OR IGNORE to skip prompts already in the DB
                data = [(p.prompt_id, p.content) for p in items]
                await db.executemany(
                    "INSERT OR IGNORE INTO prompts (prompt_id, content) VALUES (?, ?)", data
                )
                await db.commit()

    async def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._init_db()
        async with self._lock:
            placeholders = ",".join("?" for _ in ids)
            query = f"DELETE FROM prompts WHERE prompt_id IN ({placeholders})"
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(query, ids)
                await db.commit()

    async def count(self) -> int:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM prompts")
            (total,) = await cursor.fetchone()
            await cursor.close()
            return total
