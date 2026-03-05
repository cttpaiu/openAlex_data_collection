"""Async OpenAlex API client — ported from Data_Collection.ipynb."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp


class BufferedWriter:
    """Async buffered JSONL writer — minimises disk I/O."""

    def __init__(self, filename: str, buffer_size: int = 1000, mode: str = "w"):
        self.filename = filename
        self.buffer: list[str] = []
        self.buffer_size = buffer_size
        self.lock = asyncio.Lock()
        self.file = None
        self.mode = mode

    async def __aenter__(self) -> "BufferedWriter":
        self.file = open(self.filename, self.mode, encoding="utf-8", buffering=8192 * 4)
        return self

    async def __aexit__(self, *_) -> None:
        await self.flush()
        if self.file:
            self.file.close()

    async def write(self, line: str) -> None:
        async with self.lock:
            self.buffer.append(line)
            if len(self.buffer) >= self.buffer_size:
                await self._flush_locked()

    async def flush(self) -> None:
        async with self.lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if self.buffer and self.file:
            self.file.write("\n".join(self.buffer) + "\n")
            self.buffer.clear()


class AsyncOpenAlexClient:
    """Async HTTP client for OpenAlex API with retry/backoff."""

    SELECT_FIELDS = (
        "id,doi,title,publication_year,publication_date,type,"
        "primary_location,open_access,cited_by_count,"
        "citation_normalized_percentile,fwci,primary_topic,"
        "authorships,institutions_distinct_count,countries_distinct_count,"
        "updated_date,topics,abstract_inverted_index"
    )

    def __init__(self, api_key: str, email: str, per_page: int = 200,
                 max_retries: int = 5, retry_delay: int = 2,
                 concurrent_requests: int = 10):
        self.api_key = api_key
        self.email = email
        self.per_page = per_page
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncOpenAlexClient":
        headers = {
            "User-Agent": f"mailto:{self.email}",
            "api_key": self.api_key,
        }
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(
            headers=headers, timeout=timeout, connector=connector
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self.session:
            await self.session.close()

    async def get_total_count(self, api_filter: str) -> int:
        """Fire one request to get the total result count. Fast."""
        url = "https://api.openalex.org/works"
        params = {"filter": api_filter, "per_page": 1}
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data["meta"]["count"]
        except Exception:
            return 0

    async def fetch_page(self, api_filter: str, cursor: str = "*",
                         extra_params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch one cursor-paginated page of results."""
        url = "https://api.openalex.org/works"
        params: Dict[str, Any] = {
            "filter": api_filter,
            "per_page": self.per_page,
            "cursor": cursor,
            "select": self.SELECT_FIELDS,
        }
        if extra_params:
            params.update(extra_params)

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        if response.status == 403:
                            raise PermissionError("API key rejected (HTTP 403).")
                        response.raise_for_status()
                        return await response.json()
                except PermissionError:
                    raise
                except Exception:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
        return None

    async def fetch_group_by(self, api_filter: str, group_by: str,
                             cursor: str = "*") -> Optional[Dict]:
        """Fetch group_by results (topic counts etc.) using cursor pagination."""
        url = "https://api.openalex.org/works"
        params = {
            "filter": api_filter,
            "group_by": group_by,
            "per_page": 200,
            "cursor": cursor,
        }
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 429:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        response.raise_for_status()
                        return await response.json()
                except Exception:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
        return None

    async def fetch_topic_details(self, topic_id: str) -> Optional[Dict]:
        """Get topic display_name and description."""
        url = f"https://api.openalex.org/topics/{topic_id}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            pass
        return None

    async def fetch_all_pages(self, api_filter: str,
                              fields: Optional[str] = None) -> List[Dict]:
        """Collect ALL results using cursor pagination. Use carefully for large queries."""
        results: List[Dict] = []
        cursor = "*"
        extra = {"select": fields} if fields else None

        while cursor:
            data = await self.fetch_page(api_filter, cursor, extra)
            if not data:
                break
            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
            cursor = data["meta"].get("next_cursor")

        return results
