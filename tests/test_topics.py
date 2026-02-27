"""
Regression tests for topic fetching and enrichment.

BUG HISTORY
-----------
BUG-003: get-topics only returned 200 topics maximum
    Symptom : openalex get-topics --details --csv returned exactly 200 topics
              even when more topics existed in the results.
    Root cause: fetch_group_by() used page-based pagination (page=1, 2, 3...)
                but OpenAlex group_by endpoint requires cursor-based pagination.
                API returns max 200 groups per page.
    Fix: Changed fetch_group_by() to use cursor parameter and updated
         _fetch_all_groups() to iterate via meta.next_cursor.
    Test: test_fetch_all_groups_uses_cursor_pagination
          test_fetch_all_groups_paginates_multiple_pages

FEATURE-001: Topic CSV includes description and metadata
    Enhancement: CSV output now includes description, keywords, domain,
                 field, and subfield for each topic.
    Test: test_enrich_topic_data_fetches_full_details
          test_save_to_csv_includes_all_fields
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openalex.commands.topics import (
    _fetch_all_groups,
    _enrich_topic_data,
    _save_to_csv,
)


# ---------------------------------------------------------------------------
# BUG-003 — Cursor pagination for group_by
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Create a minimal config mock for topic commands."""
    cfg = MagicMock()
    cfg.api_key = "test_key"
    cfg.email = "test@example.com"
    cfg.max_retries = 3
    cfg.retry_delay = 1
    return cfg


@pytest.mark.asyncio
async def test_fetch_all_groups_uses_cursor_pagination(mock_config):
    """BUG-003: _fetch_all_groups must use cursor-based pagination, not page-based."""
    # Mock API response with cursor pagination
    mock_response_page1 = {
        "group_by": [
            {"key": "T10020", "count": 100},
            {"key": "T10682", "count": 50},
        ],
        "meta": {"next_cursor": "cursor_abc123"},
    }
    mock_response_page2 = {
        "group_by": [
            {"key": "T10382", "count": 30},
        ],
        "meta": {"next_cursor": None},  # No more pages
    }

    with patch("openalex.commands.topics.AsyncOpenAlexClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.fetch_group_by.side_effect = [
            mock_response_page1,
            mock_response_page2,
        ]
        mock_client_class.return_value = mock_client

        result = await _fetch_all_groups(mock_config, "test_filter")

        # Should have fetched both pages
        assert mock_client.fetch_group_by.call_count == 2
        
        # Verify cursor was used, not page
        calls = mock_client.fetch_group_by.call_args_list
        assert calls[0][1]["cursor"] == "*"  # First call uses *
        assert calls[1][1]["cursor"] == "cursor_abc123"  # Second call uses next_cursor
        
        # Should NOT have used page parameter
        for call in calls:
            assert "page" not in call[1], "Must use cursor, not page parameter"
        
        # Results should be combined and sorted by count descending
        assert len(result) == 3
        assert result[0]["key"] == "T10020"  # Highest count first
        assert result[1]["key"] == "T10682"
        assert result[2]["key"] == "T10382"


@pytest.mark.asyncio
async def test_fetch_all_groups_handles_empty_response(mock_config):
    """_fetch_all_groups handles empty API responses gracefully."""
    with patch("openalex.commands.topics.AsyncOpenAlexClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        # Use AsyncMock for the coroutine method
        mock_client.fetch_group_by = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await _fetch_all_groups(mock_config, "test_filter")

        assert result == []


@pytest.mark.asyncio
async def test_fetch_all_groups_handles_empty_group_by(mock_config):
    """_fetch_all_groups handles empty group_by array."""
    with patch("openalex.commands.topics.AsyncOpenAlexClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.fetch_group_by = AsyncMock(return_value={"group_by": [], "meta": {}})
        mock_client_class.return_value = mock_client

        result = await _fetch_all_groups(mock_config, "test_filter")

        assert result == []


# ---------------------------------------------------------------------------
# FEATURE-001 — Topic enrichment with description and metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enrich_topic_data_fetches_full_details(mock_config):
    """FEATURE-001: _enrich_topic_data fetches description, keywords, and hierarchy."""
    groups = [
        {"key": "T10020", "count": 100},
        {"key": "T10682", "count": 50},
    ]
    
    mock_topic_details = {
        "display_name": "Quantum Information",
        "description": "This topic covers quantum information theory...",
        "keywords": ["Quantum", "Information", "Cryptography"],
        "domain": {"id": 4, "display_name": "Physical Sciences"},
        "field": {"id": 27, "display_name": "Physics"},
        "subfield": {"id": 271, "display_name": "Quantum Physics"},
    }

    with patch("openalex.commands.topics.AsyncOpenAlexClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.fetch_topic_details.return_value = mock_topic_details
        mock_client_class.return_value = mock_client

        result = await _enrich_topic_data(mock_config, groups)

        # Verify API was called for each topic
        assert mock_client.fetch_topic_details.call_count == 2
        
        # Verify all fields are populated
        for topic in result:
            assert topic["display_name"] == "Quantum Information"
            assert topic["description"] == "This topic covers quantum information theory..."
            assert topic["keywords"] == ["Quantum", "Information", "Cryptography"]
            assert topic["domain"]["display_name"] == "Physical Sciences"
            assert topic["field"]["display_name"] == "Physics"
            assert topic["subfield"]["display_name"] == "Quantum Physics"


@pytest.mark.asyncio
async def test_enrich_topic_data_handles_missing_details(mock_config):
    """_enrich_topic_data handles topics where API returns no details."""
    groups = [{"key": "T99999", "count": 10}]

    with patch("openalex.commands.topics.AsyncOpenAlexClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.fetch_topic_details.return_value = None
        mock_client_class.return_value = mock_client

        result = await _enrich_topic_data(mock_config, groups)

        # Should have fallback empty values
        topic = result[0]
        assert topic["display_name"] == "T99999"
        assert topic["description"] == ""
        assert topic["keywords"] == []
        assert topic["domain"] == {}


# ---------------------------------------------------------------------------
# FEATURE-001 — CSV output with all fields
# ---------------------------------------------------------------------------

def test_save_to_csv_includes_all_fields():
    """FEATURE-001: CSV must include description, keywords, domain, field, subfield."""
    groups = [
        {
            "key": "https://openalex.org/T10020",
            "count": 28642,
            "display_name": "Quantum Information",
            "description": "This topic covers...",
            "keywords": ["Quantum", "Information"],
            "domain": {"display_name": "Physical Sciences"},
            "field": {"display_name": "Physics"},
            "subfield": {"display_name": "Quantum Physics"},
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
    
    try:
        _save_to_csv(groups, total_papers=100000, output=temp_path)
        
        # Read back and verify
        with open(temp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Verify header fields
            expected_fields = [
                "topic_id", "display_name", "description", "keywords",
                "domain", "field", "subfield", "paper_count", "percentage"
            ]
            assert list(rows[0].keys()) == expected_fields
            
            # Verify data
            row = rows[0]
            assert row["topic_id"] == "https://openalex.org/T10020"
            assert row["display_name"] == "Quantum Information"
            assert row["description"] == "This topic covers..."
            assert row["keywords"] == "Quantum; Information"  # Semicolon-separated
            assert row["domain"] == "Physical Sciences"
            assert row["field"] == "Physics"
            assert row["subfield"] == "Quantum Physics"
            assert row["paper_count"] == "28642"
            assert row["percentage"] == "28.64"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_save_to_csv_handles_empty_metadata():
    """CSV export handles topics with missing description/keywords gracefully."""
    groups = [
        {
            "key": "T99999",
            "count": 100,
            "display_name": "Unknown Topic",
            "description": "",
            "keywords": [],
            "domain": {},
            "field": {},
            "subfield": {},
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
    
    try:
        _save_to_csv(groups, total_papers=10000, output=temp_path)
        
        with open(temp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            row = rows[0]
            assert row["description"] == ""
            assert row["keywords"] == ""
            assert row["domain"] == ""
            assert row["field"] == ""
            assert row["subfield"] == ""
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_save_to_csv_percentage_calculation():
    """CSV percentage is calculated correctly."""
    groups = [
        {"key": "T1", "count": 2500, "display_name": "A", "description": "", "keywords": [], "domain": {}, "field": {}, "subfield": {}},
        {"key": "T2", "count": 7500, "display_name": "B", "description": "", "keywords": [], "domain": {}, "field": {}, "subfield": {}},
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
    
    try:
        _save_to_csv(groups, total_papers=10000, output=temp_path)
        
        with open(temp_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert rows[0]["percentage"] == "25.0"
            assert rows[1]["percentage"] == "75.0"
    finally:
        Path(temp_path).unlink(missing_ok=True)
