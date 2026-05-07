from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openalex.anchors import AnchorCheckResult, AnchorEntry, check_anchor_coverage, normalize_doi, normalize_title, parse_anchor_entries
from openalex.commands.check_anchor import enforce_anchor_coverage
from openalex.commands.search import search_command, search_filtered_command
from openalex.config import init_command


def test_normalize_doi_accepts_url_and_plain():
    assert normalize_doi("10.1038/NATURE23474") == "10.1038/nature23474"
    assert normalize_doi("https://doi.org/10.1038/nphys1170") == "10.1038/nphys1170"
    assert normalize_doi("doi:10.1000/182") == "10.1000/182"


def test_normalize_title_is_case_and_punctuation_insensitive():
    a = normalize_title("Quantum supremacy: using a programmable superconducting processor")
    b = normalize_title(" quantum   SUPREMACY using a programmable superconducting processor ")
    assert a == b


def test_parse_anchor_entries_splits_valid_and_invalid():
    anchors, invalid = parse_anchor_entries([
        "10.1038/nphys1170",
        "https://doi.org/10.1038/nature23474",
        "Quantum supremacy using a programmable superconducting processor",
        "doi:bad_doi",
    ])

    assert len(anchors) == 3
    assert {a.kind for a in anchors} == {"doi", "title"}
    assert invalid == ["doi:bad_doi"]


@pytest.mark.asyncio
async def test_check_anchor_coverage_marks_found_and_missing():
    cfg = SimpleNamespace(
        api_key="test",
        email="test@example.com",
        per_page=200,
        max_retries=3,
        retry_delay=1,
        concurrent_requests=5,
    )
    anchors = [
        AnchorEntry(raw="10.1038/nphys1170", kind="doi", normalized="10.1038/nphys1170"),
        AnchorEntry(
            raw="Quantum supremacy using a programmable superconducting processor",
            kind="title",
            normalized="quantum supremacy using a programmable superconducting processor",
        ),
        AnchorEntry(raw="Missing title", kind="title", normalized="missing title"),
    ]

    doi_client = AsyncMock()
    doi_client.__aenter__.return_value = doi_client
    doi_client.get_total_count = AsyncMock(return_value=1)

    title_client = AsyncMock()
    title_client.__aenter__.return_value = title_client
    # New impl fires one fetch_page per title anchor (not a full crawl).
    # Two title anchors → two side_effect responses.
    title_client.fetch_page = AsyncMock(side_effect=[
        {
            "results": [{"title": "Quantum supremacy, using a programmable superconducting processor"}],
            "meta": {"next_cursor": None},
        },
        {
            "results": [],
            "meta": {"next_cursor": None},
        },
    ])

    with patch("openalex.anchors.AsyncOpenAlexClient", side_effect=[doi_client, title_client]):
        result = await check_anchor_coverage(cfg, "base_filter", anchors)

    found_raw = {a.raw for a in result.found}
    missing_raw = {a.raw for a in result.missing}
    assert "10.1038/nphys1170" in found_raw
    assert "Quantum supremacy using a programmable superconducting processor" in found_raw
    assert "Missing title" in missing_raw


def test_init_creates_anchor_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_command.callback(force=False)

    anchor_file = tmp_path / "config" / "anchor.txt"
    assert anchor_file.exists()
    assert "One anchor paper per line" in anchor_file.read_text(encoding="utf-8")


def test_enforce_anchor_coverage_exits_on_missing():
    cfg = SimpleNamespace(
        anchor_file="config/anchor.txt",
        get_anchors=lambda: ["10.1038/nphys1170"],
    )
    parsed = [AnchorEntry(raw="10.1038/nphys1170", kind="doi", normalized="10.1038/nphys1170")]

    with (
        patch("openalex.commands.check_anchor.parse_anchor_entries", return_value=(parsed, [])),
        patch(
            "openalex.commands.check_anchor.check_anchor_coverage",
            AsyncMock(return_value=AnchorCheckResult(found=[], missing=parsed, invalid=[])),
        ),
    ):
        with pytest.raises(SystemExit):
            enforce_anchor_coverage(cfg, "base_filter", context_name="Keyword search")


def test_search_command_exits_when_anchor_check_fails():
    cfg = SimpleNamespace(
        validate_api_key=lambda: None,
        get_keywords=lambda: '"quantum computing"',
        date_from="2003-01-01",
        date_to="2024-12-31",
        doc_types=["article"],
        keywords_file="config/keywords.txt",
    )

    async def fake_get_count(*_args, **_kwargs):
        return 10

    with (
        patch("openalex.commands.search.load_config", return_value=cfg),
        patch("openalex.commands.search.check_and_print_keyword_errors", return_value=True),
        patch("openalex.commands.search.build_filter", return_value="base_filter"),
        patch("openalex.commands.search._get_count", fake_get_count),
        patch("openalex.commands.search.print_search_result_panel"),
        patch("openalex.commands.search.enforce_anchor_coverage", side_effect=SystemExit(1)),
    ):
        with pytest.raises(SystemExit):
            search_command.callback("config/collection.yml", False)


def test_search_filtered_command_exits_when_anchor_check_fails():
    cfg = SimpleNamespace(
        validate_api_key=lambda: None,
        get_keywords=lambda: '"quantum computing"',
        get_topics=lambda: ["T10020"],
        date_from="2003-01-01",
        date_to="2024-12-31",
        doc_types=["article"],
        keywords_file="config/keywords.txt",
        topics_file="config/topics.txt",
    )

    async def fake_get_count(*_args, **_kwargs):
        return 10

    with (
        patch("openalex.commands.search.load_config", return_value=cfg),
        patch("openalex.commands.search.check_and_print_keyword_errors", return_value=True),
        patch("openalex.commands.search.validate_topic_format", return_value=True),
        patch("openalex.commands.search.build_filter", return_value="base_filter"),
        patch("openalex.commands.search._get_count", fake_get_count),
        patch("openalex.commands.search.print_search_result_panel"),
        patch("openalex.commands.search.enforce_anchor_coverage", side_effect=SystemExit(1)),
    ):
        with pytest.raises(SystemExit):
            search_filtered_command.callback("config/collection.yml", False)
