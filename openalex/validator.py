"""Validation helpers for keywords and topic IDs."""

from __future__ import annotations

import re
from typing import Any

import aiohttp
from rich.console import Console

console = Console()

_TOPIC_RE = re.compile(r"^T\d{5}$")


def validate_keywords(query: str) -> list[str]:
    """Return list of error messages (empty = valid)."""
    errors: list[str] = []
    q = query.strip()

    if not q:
        errors.append("Keywords query is empty.")
        return errors

    if q.count("(") != q.count(")"):
        errors.append(
            f"Unbalanced parentheses: {q.count('(')} open, {q.count(')')} close."
        )

    if q.count('"') % 2 != 0:
        errors.append("Odd number of double quotes — every \" must have a closing \".")

    if re.search(r"\b(and|or|not)\b", q):
        errors.append("Operators must be uppercase: use OR, AND, NOT.")

    # AND NOT / OR NOT are valid Boolean patterns; only flag true duplicates
    if re.search(r"\b(AND|OR)\s+(AND|OR)\b|\bNOT\s+NOT\b", q):
        errors.append("Adjacent operators found (e.g. 'OR OR') — check query.")

    if "()" in q:
        errors.append("Empty parentheses () found.")

    return errors


def validate_topic_format(topic_id: str) -> bool:
    """Check topic ID format: T followed by exactly 5 digits."""
    return bool(_TOPIC_RE.match(topic_id))


async def validate_topics_exist(
    topic_ids: list[str],
    email: str,
    api_key: str,
) -> dict[str, bool]:
    """Check all topic IDs against the API in parallel. Returns {id: exists}."""
    results: dict[str, bool] = {}

    async def check_one(session: aiohttp.ClientSession, tid: str) -> None:
        url = f"https://api.openalex.org/topics/{tid}"
        try:
            async with session.get(url) as r:
                results[tid] = r.status == 200
        except Exception:
            results[tid] = False

    import asyncio

    headers: dict[str, Any] = {"User-Agent": f"mailto:{email}"}
    if api_key and api_key != "YOUR_API_KEY_HERE":
        headers["api_key"] = api_key

    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        await asyncio.gather(*(check_one(session, tid) for tid in topic_ids))

    return results


def check_and_print_keyword_errors(keywords: str) -> bool:
    """Print keyword errors to console. Returns True if valid."""
    errors = validate_keywords(keywords)
    if errors:
        console.print("[bold red]✗ Keyword validation failed:[/bold red]")
        for e in errors:
            console.print(f"  [red]• {e}[/red]")
        return False
    return True
