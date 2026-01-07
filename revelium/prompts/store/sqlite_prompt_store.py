from typing import List, Optional
import aiosqlite
import asyncio
from datetime import datetime
from revelium.prompts.types import Prompt
from revelium.prompts.store.prompt_store import PromptStore

class AsyncSQLitePromptStore(PromptStore):
    """Async SQLite PromptStore supporting timestamps and optional cluster_id filtering."""

    def __init__(self, db_path: str = "prompts.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._init_done = False

    async def _init_db(self):
        if self._init_done:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    cluster_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT
                )
            """)
            await db.commit()
        self._init_done = True

    async def get(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        cluster_id: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        order_by: str = "prompt_id",        # default ordering
        ascending: bool = True  
    ) -> List[Prompt]:
        """Retrieve prompts with optional pagination, cluster, and timestamp filters."""
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT prompt_id, content, cluster_id, created_at, updated_at FROM prompts WHERE 1=1"
            params = []

            if cluster_id is not None:
                query += " AND cluster_id = ?"
                params.append(cluster_id)

            if created_after:
                query += " AND created_at >= ?"
                params.append(created_after.isoformat())
            if created_before:
                query += " AND created_at <= ?"
                params.append(created_before.isoformat())
            if updated_after:
                query += " AND updated_at >= ?"
                params.append(updated_after.isoformat())
            if updated_before:
                query += " AND updated_at <= ?"
                params.append(updated_before.isoformat())
            
            order_clause = f" ORDER BY {order_by} {'ASC' if ascending else 'DESC'}"
            query += order_clause
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()

            return [
                Prompt(
                    prompt_id=row[0],
                    content=row[1],
                    cluster_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]) if row[4] else None
                )
                for row in rows
            ]
    async def update(self, items: List[Prompt]) -> None:
        if not items:
            return
        await self._init_db()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                data = [
                    (
                        p.content,
                        p.cluster_id,
                        p.updated_at.isoformat() if p.updated_at else datetime.utcnow().isoformat(),
                        p.prompt_id,
                    )
                    for p in items
                ]
                await db.executemany(
                    """
                    UPDATE prompts
                    SET content = ?,
                        cluster_id = ?,
                        updated_at = ?
                    WHERE prompt_id = ?
                    """,
                    data,
                )
                await db.commit()


    async def get_by_ids(self, ids: List[str]) -> List[Prompt]:
        if not ids:
            return []
        await self._init_db()
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT prompt_id, content, cluster_id, created_at, updated_at FROM prompts WHERE prompt_id IN ({placeholders})"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, ids)
            rows = await cursor.fetchall()
            await cursor.close()
            return [
                Prompt(
                    prompt_id=row[0],
                    content=row[1],
                    cluster_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]) if row[4] else None
                )
                for row in rows
            ]

    async def add(self, items: List[Prompt]) -> None:
        if not items:
            return
        await self._init_db()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                data = [
                    (
                        p.prompt_id,
                        p.content,
                        p.cluster_id,
                        p.created_at.isoformat(),
                        p.updated_at.isoformat() if p.updated_at else None,
                    )
                    for p in items
                ]
                await db.executemany(
                    """
                    INSERT OR IGNORE INTO prompts
                    (prompt_id, content, cluster_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    data,
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
