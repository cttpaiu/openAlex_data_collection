"""Async OpenAlex API client — ported from Data_Collection.ipynb."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from rich.console import Console

console = Console()


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
    """Async HTTP client for OpenAlex API with retry/backoff and key rotation."""

    SELECT_FIELDS = (
        "id,doi,title,publication_year,publication_date,type,"
        "primary_location,open_access,cited_by_count,"
        "citation_normalized_percentile,fwci,primary_topic,"
        "authorships,institutions_distinct_count,countries_distinct_count,"
        "updated_date,topics,abstract_inverted_index"
    )

    # Class-level variables to persist key rotation state across all instances
    _global_key_index = 0
    _global_rotation_lock = asyncio.Lock()

    def __init__(self, api_keys: list[str], email: str, per_page: int = 200,
                 max_retries: int = 5, retry_delay: int = 2,
                 concurrent_requests: int = 10):
        self.api_keys = api_keys
        self.email = email
        self.per_page = per_page
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: Optional[aiohttp.ClientSession] = None

    def _get_current_key(self) -> str:
        # Ensure index is within bounds in case keys changed
        if AsyncOpenAlexClient._global_key_index >= len(self.api_keys):
            AsyncOpenAlexClient._global_key_index = 0
        return self.api_keys[AsyncOpenAlexClient._global_key_index]

    async def _rotate_key(self) -> bool:
        """Switch to the next API key if available."""
        async with AsyncOpenAlexClient._global_rotation_lock:
            # Re-check inside lock to prevent race condition
            current = AsyncOpenAlexClient._global_key_index
            if current < len(self.api_keys) - 1:
                AsyncOpenAlexClient._global_key_index += 1
                new_key = self.api_keys[AsyncOpenAlexClient._global_key_index]
                console.print(f"[yellow]⚠ API limit reached or key rejected. Rotating to key #{AsyncOpenAlexClient._global_key_index + 1}...[/yellow]")
                if self.session:
                    self.session.headers.update({"api_key": new_key})
                return True
            return False

    async def __aenter__(self) -> "AsyncOpenAlexClient":
        headers = {
            "User-Agent": f"mailto:{self.email}",
            "api_key": self._get_current_key(),
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

    async def _request(self, method: str, url: str, **kwargs) -> Any:
        """Internal request handler with key rotation and retries."""
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "json" not in content_type.lower():
                        # OpenAlex returns HTML for URL-too-long / gateway errors.
                        # Surface a clean error instead of a JSON decode stack trace.
                        body = await response.text()
                        snippet = body.strip().replace("\n", " ")[:200]
                        full_url = str(response.url)
                        raise RuntimeError(
                            f"OpenAlex returned non-JSON response "
                            f"(HTTP {response.status}, Content-Type={content_type}). "
                            f"URL length={len(full_url)} chars. "
                            f"Likely cause: URL too long — shrink the keyword/topic/DOI filter. "
                            f"Body: {snippet}"
                        )

                    data = await response.json()

                    # Check for rate limit / budget error in JSON
                    if response.status == 403 or (isinstance(data, dict) and "error" in data and "Insufficient budget" in data.get("message", "")):
                        if await self._rotate_key():
                            # Re-try with new key immediately on rotation
                            return await self._request(method, url, **kwargs)
                        else:
                            if response.status == 403:
                                raise PermissionError("All API keys exhausted or rejected (HTTP 403).")
                            else:
                                raise RuntimeError(f"OpenAlex API Error: {data.get('message', 'Insufficient budget')}")

                    if response.status == 429:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue

                    response.raise_for_status()
                    return data
            except (PermissionError, RuntimeError):
                # Structural errors — retrying won't help.
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise e
        return None

    async def get_total_count(self, api_filter: str) -> int:
        """Fire one request to get the total result count. Fast."""
        url = "https://api.openalex.org/works"
        params = {"filter": api_filter, "per_page": 1}
        data = await self._request("GET", url, params=params)
        if data and "meta" in data:
            return data["meta"]["count"]
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
            return await self._request("GET", url, params=params)

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
            return await self._request("GET", url, params=params)

    async def fetch_topic_details(self, topic_id: str) -> Optional[Dict]:
        """Get topic display_name and description."""
        url = f"https://api.openalex.org/topics/{topic_id}"
        try:
            return await self._request("GET", url)
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
