"""Unit tests for openalex/pdf.py helpers."""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openalex.pdf import (
    _arxiv_id_from_doi,
    _cache_path,
    _normalize_doi,
    download_pdf,
    extract_first_page_text,
    find_pdf_url,
    text_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.1038/foo", "10.1038/foo"),
        ("  10.1038/foo  ", "10.1038/foo"),
        ("https://doi.org/10.1038/foo", "10.1038/foo"),
        ("http://doi.org/10.1038/foo", "10.1038/foo"),
        ("doi:10.1038/foo", "10.1038/foo"),
        ("DOI:10.1038/foo", "10.1038/foo"),
        ("HTTPS://doi.org/10.1038/foo", "10.1038/foo"),
        ("", ""),
    ],
)
def test_normalize_doi_strips_url_and_doi_prefixes(raw, expected):
    assert _normalize_doi(raw) == expected


@pytest.mark.parametrize(
    "doi,expected",
    [
        ("10.48550/arXiv.1706.07415", "1706.07415"),
        ("10.48550/ARXIV.1706.07415", "1706.07415"),
        ("10.48550/arxiv.cond-mat/0408050", "cond-mat/0408050"),
        ("10.1038/nature23474", None),
        ("10.1103/PhysRevLett.119.180509", None),
        ("", None),
    ],
)
def test_arxiv_id_from_doi(doi, expected):
    assert _arxiv_id_from_doi(doi) == expected


def test_cache_path_is_content_addressed(tmp_path):
    p1 = _cache_path("10.1038/foo", tmp_path)
    p2 = _cache_path("10.1038/foo", tmp_path)
    p3 = _cache_path("10.1038/bar", tmp_path)
    assert p1 == p2
    assert p1 != p3
    assert p1.parent == tmp_path
    assert p1.suffix == ".pdf"
    assert len(p1.stem) == 40  # SHA-1 hex


def test_text_metrics_counts_chars_and_lines():
    assert text_metrics("") == {"chars": 0, "lines": 0, "empty": True}
    assert text_metrics("abc") == {"chars": 3, "lines": 1, "empty": False}
    assert text_metrics("a\nb\nc") == {"chars": 5, "lines": 3, "empty": False}


# ─────────────────────────────────────────────────────────────────────────────
# find_pdf_url with mocked aiohttp session
# ─────────────────────────────────────────────────────────────────────────────


class _FakeResp:
    """Minimal async context manager mimicking aiohttp's response."""

    def __init__(self, status: int = 200, json_payload: dict | None = None,
                 headers: dict | None = None, body: bytes = b""):
        self.status = status
        self._json = json_payload or {}
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def json(self):
        return self._json

    @property
    def content(self):
        return _FakeContent(self._body)


class _FakeContent:
    def __init__(self, body: bytes):
        self._body = body

    async def iter_chunked(self, _size: int):
        yield self._body


def _session_returning(*ordered_responses: _FakeResp) -> MagicMock:
    """Build a fake aiohttp ClientSession whose .get(...) yields the next response."""
    it = iter(ordered_responses)
    session = MagicMock()

    def _get(*args, **kwargs):
        return next(it)

    session.get = MagicMock(side_effect=_get)
    return session


def test_find_pdf_url_unpaywall_best_oa_location_wins():
    session = _session_returning(
        _FakeResp(json_payload={
            "best_oa_location": {"url_for_pdf": "https://oa/best.pdf"},
            "oa_locations": [{"url_for_pdf": "https://oa/other.pdf"}],
        })
    )
    url, src = asyncio.run(
        find_pdf_url(session, "10.1038/foo", "me@example.com")
    )
    assert url == "https://oa/best.pdf"
    assert src == "unpaywall"


def test_find_pdf_url_unpaywall_falls_through_to_other_locations():
    session = _session_returning(
        _FakeResp(json_payload={
            "best_oa_location": {},
            "oa_locations": [
                {"url_for_pdf": None},
                {"url_for_pdf": "https://oa/second.pdf"},
            ],
        })
    )
    url, src = asyncio.run(
        find_pdf_url(session, "10.1038/foo", "me@example.com")
    )
    assert url == "https://oa/second.pdf"
    assert src == "unpaywall"


def test_find_pdf_url_falls_back_to_arxiv_for_arxiv_dois():
    session = _session_returning(
        _FakeResp(json_payload={"best_oa_location": {}, "oa_locations": []})
    )
    url, src = asyncio.run(
        find_pdf_url(session, "10.48550/arXiv.1706.07415", "me@example.com")
    )
    assert url == "https://arxiv.org/pdf/1706.07415.pdf"
    assert src == "arxiv"


def test_find_pdf_url_returns_none_when_no_source_matches():
    session = _session_returning(
        _FakeResp(json_payload={"best_oa_location": {}, "oa_locations": []})
    )
    url, src = asyncio.run(
        find_pdf_url(session, "10.1038/foo", "me@example.com")
    )
    assert (url, src) == (None, None)


def test_find_pdf_url_unpaywall_http_error_is_treated_as_no_url():
    session = _session_returning(_FakeResp(status=503))
    url, src = asyncio.run(
        find_pdf_url(session, "10.1038/foo", "me@example.com",
                     sources=("unpaywall",))
    )
    assert (url, src) == (None, None)


def test_find_pdf_url_respects_sources_filter():
    """If only arxiv is requested, unpaywall is not called."""
    session = MagicMock()
    session.get = MagicMock(side_effect=AssertionError("should not be called"))
    url, src = asyncio.run(
        find_pdf_url(session, "10.48550/arXiv.1234.5", "me@example.com",
                     sources=("arxiv",))
    )
    assert (url, src) == ("https://arxiv.org/pdf/1234.5.pdf", "arxiv")


# ─────────────────────────────────────────────────────────────────────────────
# download_pdf content-type + size guards
# ─────────────────────────────────────────────────────────────────────────────


def test_download_pdf_rejects_html_content_type(tmp_path):
    session = _session_returning(
        _FakeResp(status=200, headers={"Content-Type": "text/html"},
                  body=b"<html/>")
    )
    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path)
    )
    assert path is None
    assert err == "not_pdf"


def test_download_pdf_accepts_application_pdf(tmp_path):
    session = _session_returning(
        _FakeResp(status=200, headers={"Content-Type": "application/pdf"},
                  body=b"%PDF-1.4 fake")
    )
    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path)
    )
    assert err is None
    assert path is not None and path.exists()
    assert path.read_bytes() == b"%PDF-1.4 fake"


def test_download_pdf_accepts_octet_stream(tmp_path):
    session = _session_returning(
        _FakeResp(status=200, headers={"Content-Type": "application/octet-stream"},
                  body=b"%PDF-1.4 fake")
    )
    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path)
    )
    assert err is None
    assert path is not None and path.exists()


def test_download_pdf_enforces_size_cap(tmp_path):
    session = _session_returning(
        _FakeResp(status=200, headers={"Content-Type": "application/pdf"},
                  body=b"a" * (5 * 1024))
    )
    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path,
                     max_bytes=1024)
    )
    assert path is None
    assert err == "too_large"


def test_download_pdf_handles_404(tmp_path):
    session = _session_returning(_FakeResp(status=404))
    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path)
    )
    assert path is None
    assert err == "http_404"


def test_download_pdf_uses_cache_on_second_call(tmp_path):
    cached = _cache_path("10.1/x", tmp_path)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"cached-pdf-bytes")

    # session.get should never be called when cache hits.
    session = MagicMock()
    session.get = MagicMock(side_effect=AssertionError("should not fetch"))

    path, err = asyncio.run(
        download_pdf(session, "https://x/foo.pdf", "10.1/x", tmp_path)
    )
    assert err is None
    assert path == cached
    assert path.read_bytes() == b"cached-pdf-bytes"


# ─────────────────────────────────────────────────────────────────────────────
# extract_first_page_text with a synthesized PDF
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_first_page_text_reads_text_layer(tmp_path):
    import pymupdf  # type: ignore

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Author A\nMIT, Cambridge MA")
    pdf_path = tmp_path / "fixture.pdf"
    doc.save(str(pdf_path))
    doc.close()

    text = extract_first_page_text(pdf_path)
    assert "Author A" in text
    assert "MIT, Cambridge MA" in text


def test_extract_first_page_text_returns_empty_on_corrupt_file(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not-a-pdf")
    assert extract_first_page_text(bad) == ""
