"""SQLite-based store implementation with vector search capabilities.

This store provides persistent storage using SQLite with optional vector search functionality.
It implements the BaseStore interface from langgraph.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Annotated
import uuid

import aiosqlite
from langchain_core.embeddings import Embeddings
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)
        

@tool
async def save_memory(memories: List[str], *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
    '''Save the given memory for the current user. Do not save duplicate memories.'''
    user_id = config.get("configurable", {}).get("user_id", "myself")
    namespace = ("memories", user_id)
    for memory in memories:
        id = uuid.uuid4().hex
        await store.aput(namespace, f"memory_{id}", {"data": memory})
    return f"Saved memories: {memories}"

async def get_memories(store: BaseStore, user_id: str = "myself", query: str = None) -> List[str]:
    namespace = ("memories", user_id)
    memories = [m.value["data"] for m in await store.asearch(namespace, query=query)]
    return memories

class SqliteStore(BaseStore):
    """SQLite-based store with optional vector search.

    This store provides persistent storage using SQLite with optional vector search functionality.
    Data is stored in two tables:
    - items: Stores the actual key-value pairs with their metadata
    - vectors: Stores vector embeddings for semantic search

    Args:
        db_path (Union[str, Path]): Path to the SQLite database file
        index (Optional[IndexConfig]): Configuration for vector search functionality
    """

    def __init__(
        self, db_path: Union[str, Path], *, index: Optional[IndexConfig] = None
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_config = index
        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings: Optional[Embeddings] = ensure_embeddings(
                self.index_config.get("embed"),
            )
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p)
                for p in (self.index_config.get("fields") or ["$"])
            ]
        else:
            self.index_config = None
            self.embeddings = None

    async def _init_db(self, db: aiosqlite.Connection) -> None:
        """Initialize database schema.

        Args:
            db (aiosqlite.Connection): Database connection
        """
        await db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                namespace TEXT,
                key TEXT,
                value TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (namespace, key)
            )
        """)
        if self.index_config:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    namespace TEXT,
                    key TEXT,
                    path TEXT,
                    vector BLOB,
                    PRIMARY KEY (namespace, key, path),
                    FOREIGN KEY (namespace, key) REFERENCES items (namespace, key)
                        ON DELETE CASCADE
                )
            """)
        await db.commit()

    def batch(self, ops: List[Op]) -> List[Result]:
        """Execute a batch of operations synchronously.

        Args:
            ops (List[Op]): List of operations to execute

        Returns:
            List[Result]: Results of the operations
        """
        raise NotImplementedError

    async def abatch(self, ops: List[Op]) -> List[Result]:
        """Execute a batch of operations asynchronously.

        Args:
            ops (List[Op]): List of operations to execute

        Returns:
            List[Result]: Results of the operations
        """
        async with aiosqlite.connect(self.db_path) as db:
            await self._init_db(db)
            results: List[Result] = []
            put_ops: Dict[Tuple[Tuple[str, ...], str], PutOp] = {}
            search_ops: Dict[int, Tuple[SearchOp, List[Tuple[Item, List[List[float]]]]]] = {}

            for i, op in enumerate(ops):
                if isinstance(op, GetOp):
                    item = await self._get_item(db, op.namespace, op.key)
                    results.append(item)
                elif isinstance(op, SearchOp):
                    candidates = await self._filter_items(db, op)
                    search_ops[i] = (op, candidates)
                    results.append(None)
                elif isinstance(op, ListNamespacesOp):
                    namespaces = await self._list_namespaces(db, op)
                    results.append(namespaces)
                elif isinstance(op, PutOp):
                    put_ops[(op.namespace, op.key)] = op
                    results.append(None)
                else:
                    raise ValueError(f"Unknown operation type: {type(op)}")

            if search_ops:
                query_vectors = await self._embed_search_queries(search_ops)
                await self._batch_search(db, search_ops, query_vectors, results)

            to_embed = self._extract_texts(put_ops)
            if to_embed and self.index_config and self.embeddings:
                embeddings = await self.embeddings.aembed_documents(list(to_embed))
                await self._insert_vectors(db, to_embed, embeddings)

            await self._apply_put_ops(db, put_ops)
            await db.commit()

            return results

    async def _get_item(
        self, db: aiosqlite.Connection, namespace: Tuple[str, ...], key: str
    ) -> Optional[Item]:
        """Get an item from the database.

        Args:
            db (aiosqlite.Connection): Database connection
            namespace (Tuple[str, ...]): Item namespace
            key (str): Item key

        Returns:
            Optional[Item]: The item if found, None otherwise
        """
        async with db.execute(
            "SELECT value, created_at, updated_at FROM items WHERE namespace = ? AND key = ?",
            ("/".join(namespace), key)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return Item(
                    namespace=namespace,
                    key=key,
                    value=json.loads(row[0]),
                    created_at=datetime.fromisoformat(row[1]),
                    updated_at=datetime.fromisoformat(row[2])
                )
            return None

    async def _filter_items(
        self, db: aiosqlite.Connection, op: SearchOp
    ) -> List[Tuple[Item, List[List[float]]]]:
        """Filter items by namespace and filter function.

        Args:
            db (aiosqlite.Connection): Database connection
            op (SearchOp): Search operation

        Returns:
            List[Tuple[Item, List[List[float]]]]: Filtered items with their vectors
        """
        namespace_prefix = "/".join(op.namespace_prefix)
        query = """
            SELECT namespace, key, value, created_at, updated_at
            FROM items
            WHERE namespace LIKE ?
        """
        params = [f"{namespace_prefix}%"]

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            filtered = []
            for row in rows:
                item = Item(
                    namespace=tuple(row[0].split("/")),
                    key=row[1],
                    value=json.loads(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4])
                )
                if not op.filter or all(
                    self._compare_values(item.value.get(key), filter_value)
                    for key, filter_value in op.filter.items()
                ):
                    if op.query and self.index_config:
                        vectors = await self._get_vectors(db, item.namespace, item.key)
                        filtered.append((item, vectors))
                    else:
                        filtered.append((item, []))
            return filtered

    async def _get_vectors(
        self, db: aiosqlite.Connection, namespace: Tuple[str, ...], key: str
    ) -> List[List[float]]:
        """Get vectors for an item.

        Args:
            db (aiosqlite.Connection): Database connection
            namespace (Tuple[str, ...]): Item namespace
            key (str): Item key

        Returns:
            List[List[float]]: List of vectors
        """
        async with db.execute(
            "SELECT vector FROM vectors WHERE namespace = ? AND key = ?",
            ("/".join(namespace), key)
        ) as cursor:
            rows = await cursor.fetchall()
            return [json.loads(row[0]) for row in rows]

    async def _list_namespaces(
        self, db: aiosqlite.Connection, op: ListNamespacesOp
    ) -> List[Tuple[str, ...]]:
        """List namespaces matching the conditions.

        Args:
            db (aiosqlite.Connection): Database connection
            op (ListNamespacesOp): List namespaces operation

        Returns:
            List[Tuple[str, ...]]: List of matching namespaces
        """
        async with db.execute("SELECT DISTINCT namespace FROM items") as cursor:
            rows = await cursor.fetchall()
            namespaces = [tuple(ns.split("/")) for (ns,) in rows]

            if op.match_conditions:
                namespaces = [
                    ns for ns in namespaces
                    if all(self._does_match(condition, ns) for condition in op.match_conditions)
                ]

            if op.max_depth is not None:
                namespaces = sorted({ns[:op.max_depth] for ns in namespaces})
            else:
                namespaces = sorted(namespaces)

            return namespaces[op.offset:op.offset + op.limit]

    async def _embed_search_queries(
        self,
        search_ops: Dict[int, Tuple[SearchOp, List[Tuple[Item, List[List[float]]]]]],
    ) -> Dict[str, List[float]]:
        """Embed search queries.

        Args:
            search_ops (Dict[int, Tuple[SearchOp, List[Tuple[Item, List[List[float]]]]]]): Search operations

        Returns:
            Dict[str, List[float]]: Query embeddings
        """
        query_vectors = {}
        if self.index_config and self.embeddings and search_ops:
            queries = {op.query for (op, _) in search_ops.values() if op.query}
            if queries:
                embeddings = await self.embeddings.aembed_documents(list(queries))
                query_vectors = dict(zip(queries, embeddings))
        return query_vectors

    async def _batch_search(
        self,
        db: aiosqlite.Connection,
        ops: Dict[int, Tuple[SearchOp, List[Tuple[Item, List[List[float]]]]]],
        query_vectors: Dict[str, List[float]],
        results: List[Result],
    ) -> None:
        """Perform batch similarity search.

        Args:
            db (aiosqlite.Connection): Database connection
            ops (Dict[int, Tuple[SearchOp, List[Tuple[Item, List[List[float]]]]]]): Search operations
            query_vectors (Dict[str, List[float]]): Query embeddings
            results (List[Result]): Results list to update
        """
        for i, (op, candidates) in ops.items():
            if not candidates:
                results[i] = []
                continue

            if op.query and query_vectors:
                query_vector = query_vectors[op.query]
                flat_items, flat_vectors = [], []
                scoreless = []

                for item, vectors in candidates:
                    for vector in vectors:
                        flat_items.append(item)
                        flat_vectors.append(vector)
                    if not vectors:
                        scoreless.append(item)

                scores = self._cosine_similarity(query_vector, flat_vectors)
                sorted_results = sorted(
                    zip(scores, flat_items), key=lambda x: x[0], reverse=True
                )

                seen = set()
                kept = []
                for score, item in sorted_results:
                    key = (item.namespace, item.key)
                    if key in seen:
                        continue
                    ix = len(seen)
                    seen.add(key)
                    if ix >= op.offset + op.limit:
                        break
                    if ix < op.offset:
                        continue
                    kept.append((score, item))

                if scoreless and len(kept) < op.limit:
                    kept.extend(
                        (None, item) for item in scoreless[:op.limit - len(kept)]
                    )

                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                        score=float(score) if score is not None else None,
                    )
                    for score, item in kept
                ]
            else:
                results[i] = [
                    SearchItem(
                        namespace=item.namespace,
                        key=item.key,
                        value=item.value,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                    for (item, _) in candidates[op.offset:op.offset + op.limit]
                ]

    async def _apply_put_ops(
        self, db: aiosqlite.Connection, put_ops: Dict[Tuple[Tuple[str, ...], str], PutOp]
    ) -> None:
        """Apply put operations to the database.

        Args:
            db (aiosqlite.Connection): Database connection
            put_ops (Dict[Tuple[Tuple[str, ...], str], PutOp]): Put operations
        """
        for (namespace, key), op in put_ops.items():
            if op.value is None:
                await db.execute(
                    "DELETE FROM items WHERE namespace = ? AND key = ?",
                    ("/".join(namespace), key)
                )
            else:
                now = datetime.now(timezone.utc)
                await db.execute(
                    """
                    INSERT INTO items (namespace, key, value, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (namespace, key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (
                        "/".join(namespace),
                        key,
                        json.dumps(op.value),
                        now.isoformat(),
                        now.isoformat(),
                    )
                )

    async def _insert_vectors(
        self,
        db: aiosqlite.Connection,
        to_embed: Dict[str, List[Tuple[Tuple[str, ...], str, str]]],
        embeddings: List[List[float]],
    ) -> None:
        """Insert vector embeddings into the database.

        Args:
            db (aiosqlite.Connection): Database connection
            to_embed (Dict[str, List[Tuple[Tuple[str, ...], str, str]]]): Texts to embed
            embeddings (List[List[float]]): Vector embeddings
        """
        indices = [index for indices in to_embed.values() for index in indices]
        if len(indices) != len(embeddings):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) does not"
                f" match number of indices ({len(indices)})"
            )

        for embedding, (ns, key, path) in zip(embeddings, indices):
            await db.execute(
                """
                INSERT INTO vectors (namespace, key, path, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (namespace, key, path) DO UPDATE SET
                    vector = excluded.vector
                """,
                ("/".join(ns), key, path, json.dumps(embedding))
            )

    def _extract_texts(
        self, put_ops: Dict[Tuple[Tuple[str, ...], str], PutOp]
    ) -> Dict[str, List[Tuple[Tuple[str, ...], str, str]]]:
        """Extract texts for embedding from put operations.

        Args:
            put_ops (Dict[Tuple[Tuple[str, ...], str], PutOp]): Put operations

        Returns:
            Dict[str, List[Tuple[Tuple[str, ...], str, str]]]: Texts to embed
        """
        if put_ops and self.index_config and self.embeddings:
            to_embed = {}
            for op in put_ops.values():
                if op.value is not None and op.index is not False:
                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for i, text in enumerate(texts):
                                    to_embed.setdefault(text, []).append(
                                        (op.namespace, op.key, f"{path}.{i}")
                                    )
                            else:
                                to_embed.setdefault(texts[0], []).append(
                                    (op.namespace, op.key, path)
                                )
            return to_embed
        return {}

    def _compare_values(self, item_value: Any, filter_value: Any) -> bool:
        """Compare values in a JSONB-like way.

        Args:
            item_value (Any): Value from the item
            filter_value (Any): Value from the filter

        Returns:
            bool: Whether the values match
        """
        if isinstance(filter_value, dict):
            if any(k.startswith("$") for k in filter_value):
                return all(
                    self._apply_operator(item_value, op_key, op_value)
                    for op_key, op_value in filter_value.items()
                )
            if not isinstance(item_value, dict):
                return False
            return all(
                self._compare_values(item_value.get(k), v)
                for k, v in filter_value.items()
            )
        elif isinstance(filter_value, (list, tuple)):
            return (
                isinstance(item_value, (list, tuple))
                and len(item_value) == len(filter_value)
                and all(
                    self._compare_values(iv, fv)
                    for iv, fv in zip(item_value, filter_value)
                )
            )
        else:
            return item_value == filter_value

    def _apply_operator(self, value: Any, operator: str, op_value: Any) -> bool:
        """Apply a comparison operator.

        Args:
            value (Any): Value to compare
            operator (str): Operator to apply
            op_value (Any): Value to compare against

        Returns:
            bool: Result of the comparison
        """
        if operator == "$eq":
            return value == op_value
        elif operator == "$gt":
            return float(value) > float(op_value)
        elif operator == "$gte":
            return float(value) >= float(op_value)
        elif operator == "$lt":
            return float(value) < float(op_value)
        elif operator == "$lte":
            return float(value) <= float(op_value)
        elif operator == "$ne":
            return value != op_value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _does_match(self, match_condition: MatchCondition, key: Tuple[str, ...]) -> bool:
        """Check if a namespace key matches a match condition.

        Args:
            match_condition (MatchCondition): Match condition to check
            key (Tuple[str, ...]): Namespace key to check

        Returns:
            bool: Whether the key matches the condition
        """
        match_type = match_condition.match_type
        path = match_condition.path

        if len(key) < len(path):
            return False

        if match_type == "prefix":
            for k_elem, p_elem in zip(key, path):
                if p_elem == "*":
                    continue
                if k_elem != p_elem:
                    return False
            return True
        elif match_type == "suffix":
            for k_elem, p_elem in zip(reversed(key), reversed(path)):
                if p_elem == "*":
                    continue
                if k_elem != p_elem:
                    return False
            return True
        else:
            raise ValueError(f"Unsupported match type: {match_type}")

    def _cosine_similarity(self, X: List[float], Y: List[List[float]]) -> List[float]:
        """Compute cosine similarity between a vector X and a matrix Y.

        Args:
            X (List[float]): Query vector
            Y (List[List[float]]): Matrix of vectors to compare against

        Returns:
            List[float]: Cosine similarities
        """
        if not Y:
            return []

        try:
            import numpy as np
            X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
            Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y
            X_norm = np.linalg.norm(X_arr)
            Y_norm = np.linalg.norm(Y_arr, axis=1)

            mask = Y_norm != 0
            similarities = np.zeros_like(Y_norm)
            similarities[mask] = np.dot(Y_arr[mask], X_arr) / (Y_norm[mask] * X_norm)
            return similarities.tolist()
        except ImportError:
            logger.warning(
                "NumPy not found. Using pure Python implementation for vector operations. "
                "This may significantly impact performance. Consider installing NumPy: "
                "pip install numpy"
            )
            similarities = []
            for y in Y:
                dot_product = sum(a * b for a, b in zip(X, y))
                norm1 = sum(a * a for a in X) ** 0.5
                norm2 = sum(a * a for a in y) ** 0.5
                similarity = (
                    dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                )
                similarities.append(similarity)
            return similarities 