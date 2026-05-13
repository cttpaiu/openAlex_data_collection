"""PDF helpers for the ``openalex impute pdf`` source.

Three pure functions, no DB / no LLM:

* :func:`find_pdf_url` — given a DOI, return a downloadable PDF URL via
  Unpaywall's polite pool, with an arXiv fallback when the DOI prefix
  identifies arXiv (``10.48550/arXiv.<id>``).
* :func:`download_pdf` — fetch the URL to a SHA1-keyed cache directory.
  Skips the download when the cached file already exists.
* :func:`extract_first_page_text` — return the first page's text layer
  via PyMuPDF. No OCR; scanned PDFs return an empty string.

The functions are async-friendly (``aiohttp``) for network operations and
synchronous for the local file I/O so callers can compose them with the
rest of the imputation async loop.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import aiohttp

UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
ARXIV_DOI_PREFIX = "10.48550/arxiv."


def _normalize_doi(raw: str) -> str:
    s = (raw or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def _arxiv_id_from_doi(doi: str) -> str | None:
    """Return the arXiv identifier when the DOI is an arXiv-minted one."""
    lower = doi.lower()
    if lower.startswith(ARXIV_DOI_PREFIX):
        return doi[len(ARXIV_DOI_PREFIX):]
    return None


async def find_pdf_url(
    session: aiohttp.ClientSession,
    doi: str,
    email: str,
    sources: tuple[str, ...] = ("unpaywall", "arxiv"),
) -> tuple[str | None, str | None]:
    """Resolve a DOI to a downloadable PDF URL.

    Returns ``(url, source)`` where source is ``"unpaywall"`` or ``"arxiv"``,
    or ``(None, None)`` when no source returned a URL.
    """
    normalized = _normalize_doi(doi)

    if "unpaywall" in sources:
        url = await _unpaywall_pdf_url(session, normalized, email)
        if url:
            return url, "unpaywall"

    if "arxiv" in sources:
        arxiv_id = _arxiv_id_from_doi(normalized)
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv"

    return None, None


async def _unpaywall_pdf_url(
    session: aiohttp.ClientSession, doi: str, email: str
) -> str | None:
    url = UNPAYWALL_URL.format(doi=doi)
    params = {"email": email}
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            payload = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None

    best = payload.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    if pdf:
        return pdf

    for loc in payload.get("oa_locations") or []:
        if isinstance(loc, dict) and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


def _cache_path(doi: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(doi.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pdf"


async def download_pdf(
    session: aiohttp.ClientSession,
    url: str,
    doi: str,
    cache_dir: Path,
    max_bytes: int = 30 * 1024 * 1024,
    skip_cache: bool = False,
) -> tuple[Path | None, str | None]:
    """Download a PDF to a content-addressed file. Returns ``(path, err)``.

    ``err`` is ``None`` on success or a short reason string on failure
    (``"too_large"``, ``"http_404"``, ``"network"``, ``"not_pdf"``).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(doi, cache_dir)
    if path.exists() and not skip_cache:
        return path, None

    try:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status == 404:
                return None, "http_404"
            if resp.status >= 400:
                return None, f"http_{resp.status}"
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "pdf" not in ctype and "application/octet-stream" not in ctype:
                # Some publishers serve PDF as octet-stream or with no
                # Content-Type. Allow octet-stream; reject HTML pages etc.
                return None, "not_pdf"
            written = 0
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as f:
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        tmp.unlink(missing_ok=True)
                        return None, "too_large"
                    f.write(chunk)
            tmp.rename(path)
            return path, None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None, "network"


def extract_first_page_text(path: Path) -> str:
    """Return the first page's text layer from a PDF. Empty if no text.

    PyMuPDF import is lazy so callers that never trigger PDF extraction
    don't pay the import cost.
    """
    import pymupdf  # type: ignore

    try:
        with pymupdf.open(str(path)) as doc:
            if doc.page_count == 0:
                return ""
            page = doc.load_page(0)
            text = page.get_text("text") or ""
            return text.strip()
    except Exception:
        return ""


def text_metrics(text: str) -> dict[str, Any]:
    """Quick diagnostic counters for extracted text."""
    return {
        "chars": len(text),
        "lines": text.count("\n") + (1 if text else 0),
        "empty": not bool(text),
    }
