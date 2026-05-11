from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
from openalex.commands.database import OpenAlexLoader
from openalex.commands.impute_country import (
    _compute_rule_inference,
    _extract_json_payload,
    _groq_chat,
    _ollama_chat,
    _parse_reset_seconds,
    _retry_wait_seconds,
)
from openalex.imputation import infer_country_from_affiliation, normalize_country_code


def test_infer_country_unambiguous():
    inf = infer_country_from_affiliation(
        "Department of Physics, University of Oxford, Oxford, United Kingdom"
    )
    assert inf.status == "unambiguous"
    assert inf.country_code == "GB"


def test_infer_country_ambiguous():
    inf = infer_country_from_affiliation(
        "Joint Center between USA and China for Quantum Systems"
    )
    assert inf.status == "ambiguous"
    assert inf.country_code is None


def test_infer_country_none():
    inf = infer_country_from_affiliation(
        "School of Geodesy and Geomatics, Wuhan Univ"
    )
    assert inf.status == "none"
    assert inf.country_code is None


def test_compute_inference_splits_statuses():
    rows = [
        (1, "W1", "MIT, Cambridge, MA, USA"),
        (2, "W2", "Joint center between USA and China"),
        (3, "W3", "Wuhan University"),
    ]
    result = _compute_rule_inference(rows)
    assert result["eligible"] == 3
    assert len(result["updates"]) == 1
    assert result["ambiguous"] == 1
    assert result["none"] == 1


def test_normalize_country_code_hong_kong_to_china():
    assert normalize_country_code("HK") == "CN"
    assert normalize_country_code("cn") == "CN"


def test_extract_json_payload_from_markdown_fence():
    payload = _extract_json_payload(
        "```json\n"
        + json.dumps({"predictions": [{"row_id": 1, "country_code": "US", "status": "unambiguous", "confidence": 0.95, "reason": "usa"}]})
        + "\n```"
    )
    assert payload["predictions"][0]["country_code"] == "US"


def test_extract_json_payload_ignores_trailing_extra_data():
    payload = _extract_json_payload('prefix {"predictions":[{"row_id":1}]} trailing {"ignored":true}')
    assert payload["predictions"][0]["row_id"] == 1


def test_retry_wait_seconds_parses_try_again_message():
    wait = _retry_wait_seconds("Please try again in 18.26s")
    assert wait == pytest.approx(18.26)


def test_parse_reset_seconds_supports_duration_strings():
    assert _parse_reset_seconds("7.66s") == pytest.approx(7.66)
    assert _parse_reset_seconds("2m59.56s") == pytest.approx(179.56)
    assert _parse_reset_seconds("1") == pytest.approx(1.0)


def test_database_loader_preserves_raw_affiliation_when_no_institution(tmp_path):
    db_path = tmp_path / "test.duckdb"
    jsonl_path = tmp_path / "dummy.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    loader = OpenAlexLoader(str(db_path), str(jsonl_path))
    try:
        loader.process_record(
            {
                "id": "https://openalex.org/W123",
                "title": "Test Paper",
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A123", "display_name": "Author A"},
                        "institutions": [],
                        "countries": [],
                        "raw_author_name": "Author A",
                        "author_position": "first",
                        "is_corresponding": True,
                        "raw_affiliation_strings": ["Department of Physics, University of Oxford, United Kingdom"],
                    }
                ],
            }
        )
        raw_aff = loader.con.execute(
            "SELECT raw_affiliation_string FROM contributions WHERE paper_id = 'W123' LIMIT 1"
        ).fetchone()[0]
        assert raw_aff == "Department of Physics, University of Oxford, United Kingdom"
    finally:
        loader.close()


def test_groq_chat_sets_browser_like_headers(monkeypatch):
    captured: dict[str, str | int] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps({"ok": True})}}
                    ]
                }
            ).encode()

    def fake_urlopen(req, timeout=60):
        headers = {k.lower(): v for k, v in req.header_items()}
        captured["user_agent"] = headers.get("user-agent", "")
        captured["accept"] = headers.get("accept", "")
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    payload = _groq_chat("gsk_test", "allam-2-7b", "Return JSON")
    assert payload["ok"] is True
    assert captured["user_agent"] == "Mozilla/5.0"
    assert captured["accept"] == "application/json"
    assert captured["timeout"] == 60


def test_groq_chat_surfaces_http_error_details(monkeypatch):
    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b"error code: 1010"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Groq HTTP 403: error code: 1010"):
        _groq_chat("gsk_test", "allam-2-7b", "Return JSON")


def test_groq_chat_retries_on_429_then_succeeds(monkeypatch):
    attempts = {"count": 0}
    sleeps: list[float] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps({"ok": True})}}
                    ]
                }
            ).encode()

    def fake_urlopen(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError(
                url="https://api.groq.com/openai/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=BytesIO(
                    b'{"error":{"message":"Rate limit reached. Please try again in 3.5s.","code":"rate_limit_exceeded"}}'
                ),
            )
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    payload = _groq_chat("gsk_test", "allam-2-7b", "Return JSON", max_retries=2)
    assert payload["ok"] is True
    assert attempts["count"] == 2
    assert sleeps and sleeps[0] == pytest.approx(3.5, rel=0.01)


def test_ollama_chat_parses_json_message(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps({"predictions": [{"row_id": 1}]})
                    }
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Resp())
    payload = _ollama_chat(
        base_url="http://localhost:11434",
        model="sorc/qwen3.5-instruct:2b",
        prompt="Return JSON",
        max_tokens=120,
    )
    assert payload["predictions"][0]["row_id"] == 1
