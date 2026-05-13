from __future__ import annotations

from openalex.commands.database import OpenAlexLoader
from openalex.commands.impute_affiliation import _compute_rule_inference
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


