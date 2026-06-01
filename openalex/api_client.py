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


class _ApiKeyPool:
    """Process-wide pool of OpenAlex API keys with per-key health tracking.

    Replaces the old single-index rotation. Each key is in one of three states:
        active     — fresh; eligible for new requests
        exhausted  — server returned "Insufficient budget" (resets at UTC midnight)
        rejected   — server returned 403 unrelated to budget (bad key / revoked)

    `acquire()` returns the next active key (round-robin). `mark()` atomically
    flips a key into a terminal state. Multiple concurrent requests that all
    fail on the same key will each call `mark()` — the first transition logs
    the rotation; the rest are idempotent no-ops. The next `acquire()` call
    then picks the next still-active key. No false "all keys exhausted" when
    sibling tasks were merely racing on a stale key.
    """

    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    REJECTED = "rejected"

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.state: list[str] = []
        self.usage: list[int] = []
        self._cursor: int = 0
        self._lock: Optional[asyncio.Lock] = None  # lazy: created in running loop

    def _lock_for_loop(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def initialize(self, keys: list[str]) -> None:
        """Install the key list. Skips wholesale replacement if same keys reload."""
        new_keys = list(keys or [])
        if new_keys == self.keys:
            return
        self.keys = new_keys
        self.state = [self.ACTIVE] * len(new_keys)
        self.usage = [0] * len(new_keys)
        self._cursor = 0

    async def acquire(self) -> tuple[int, str] | tuple[None, None]:
        async with self._lock_for_loop():
            n = len(self.keys)
            if n == 0:
                return None, None
            for offset in range(n):
                i = (self._cursor + offset) % n
                if self.state[i] == self.ACTIVE:
                    self._cursor = (i + 1) % n
                    self.usage[i] += 1
                    return i, self.keys[i]
            return None, None

    async def mark(self, idx: int, new_state: str) -> bool:
        """Mark a key with a terminal state. Returns True if state actually flipped."""
        if not (0 <= idx < len(self.keys)):
            return False
        async with self._lock_for_loop():
            if self.state[idx] != self.ACTIVE:
                return False
            self.state[idx] = new_state
            next_active = next(
                (i for i, s in enumerate(self.state) if s == self.ACTIVE),
                None,
            )
            if next_active is not None:
                console.print(
                    f"[yellow]⚠ Key #{idx + 1} {new_state}. "
                    f"Switching to key #{next_active + 1} "
                    f"(active: {self.state.count(self.ACTIVE)}/{len(self.state)}).[/yellow]"
                )
            else:
                console.print(
                    f"[red]✗ Key #{idx + 1} {new_state}. No keys left "
                    f"(exhausted: {self.state.count(self.EXHAUSTED)}, "
                    f"rejected: {self.state.count(self.REJECTED)}).[/red]"
                )
            return True

    def has_active(self) -> bool:
        return any(s == self.ACTIVE for s in self.state)

    def snapshot(self) -> dict:
        return {
            "total": len(self.keys),
            "active": self.state.count(self.ACTIVE),
            "exhausted": self.state.count(self.EXHAUSTED),
            "rejected": self.state.count(self.REJECTED),
            "usage": list(self.usage),
        }


class AsyncOpenAlexClient:
    """Async HTTP client for OpenAlex API with retry/backoff and key rotation."""

    SELECT_FIELDS = (
        "id,doi,title,publication_year,publication_date,type,"
        "primary_location,open_access,cited_by_count,"
        "citation_normalized_percentile,fwci,primary_topic,"
        "authorships,institutions_distinct_count,countries_distinct_count,"
        "updated_date,topics,abstract_inverted_index"
    )

    # Process-wide pool — survives across client instances so multiple CLI
    # subcommands in the same run share key-health state.
    _pool: _ApiKeyPool = _ApiKeyPool()

    def __init__(self, api_keys: list[str], email: str, per_page: int = 200,
                 max_retries: int = 5, retry_delay: int = 2,
                 concurrent_requests: int = 10):
        self.api_keys = list(api_keys)
        AsyncOpenAlexClient._pool.initialize(self.api_keys)
        self.email = email
        self.per_page = per_page
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncOpenAlexClient":
        # Only the User-Agent goes on the session — api_key is injected
        # per-request from the pool to avoid cross-task header races.
        headers = {"User-Agent": f"mailto:{self.email}"}
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
        """Internal request handler. Acquires a key per attempt, marks it on
        budget/403 errors, and tries the next live key. Raises PermissionError
        only when the pool truly has no active key left.
        """
        last_data: Any = None
        for attempt in range(self.max_retries):
            idx, key = await AsyncOpenAlexClient._pool.acquire()
            if idx is None:
                raise PermissionError(
                    "All API keys exhausted or rejected — "
                    "OpenAlex budget resets at UTC midnight."
                )

            local_kwargs = dict(kwargs)
            hdrs = dict(local_kwargs.get("headers") or {})
            hdrs["api_key"] = key
            local_kwargs["headers"] = hdrs

            try:
                async with self.session.request(method, url, **local_kwargs) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "json" not in content_type.lower():
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
                    last_data = data

                    msg = (
                        data.get("message", "")
                        if isinstance(data, dict)
                        else ""
                    )
                    is_budget = (
                        isinstance(data, dict)
                        and "error" in data
                        and "Insufficient budget" in msg
                    )
                    is_forbidden = response.status == 403

                    if is_budget:
                        await AsyncOpenAlexClient._pool.mark(idx, _ApiKeyPool.EXHAUSTED)
                        continue
                    if is_forbidden:
                        await AsyncOpenAlexClient._pool.mark(idx, _ApiKeyPool.REJECTED)
                        continue

                    if response.status == 429:
                        # Rate-limited but the key is still good — back off and
                        # retry with the same (or another active) key.
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue

                    response.raise_for_status()
                    return data
            except (PermissionError, RuntimeError):
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise e

        # Loop ended without success and without an explicit raise.
        if not AsyncOpenAlexClient._pool.has_active():
            raise PermissionError(
                "All API keys exhausted or rejected — "
                "OpenAlex budget resets at UTC midnight."
            )
        raise RuntimeError(
            f"OpenAlex request failed after {self.max_retries} attempts. "
            f"Last response: {last_data!r}"
        )

    @classmethod
    def pool_snapshot(cls) -> dict:
        return cls._pool.snapshot()

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
