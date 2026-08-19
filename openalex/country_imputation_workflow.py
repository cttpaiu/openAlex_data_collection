"""
Repeatable Country Imputation Workflow
========================================
Derived from the OpenAlex weather-tech-papers pipeline (contributions table).

Recovers null country_code values in a `contributions` table using four
passes, in order of precision/reliability (most trustworthy first):
  0. ROR API institution repair — fixes institutions.country_code itself
     by looking up each institution's ror_id against the LIVE ROR API.
     This is a data REPAIR, not a guess (exact ID match), and should
     always run first since it feeds Pass A. Best-performing method
     found so far: ~99% hit rate on a real test (102/103 institutions).
  1. Join-fill from a matched institution_id (free win — now much more
     productive once Pass 0 has repaired institutions.country_code)
  2. Country-name keyword match (via pycountry, NOT a possibly-corrupt
     local `countries` table)
  3. Sub-national region matching (US states, Indian states — extend this
     list with other countries' states/provinces as needed per dataset)

Design principles carried over from the original session:
  - Never trust a table's own name column blindly — pycountry is the
    reliable source for real country names.
  - Sort candidate names longest-first to avoid partial-substring
    false matches (e.g. "Guinea" inside "Equatorial Guinea").
  - Always spot-check a random sample of matches before trusting a pass.
  - Ambiguous names that double as other things (e.g. "Georgia" the US
    state vs. the country) should be excluded or handled with an extra
    guard rule, not matched blindly.
  - Exact-ID lookups (Pass 0) don't need manual review the way text-based
    matching does — there's no ambiguity in an ID match, so it auto-writes.

Usage (as a library):
    from openalex.imputation.country import run_country_imputation
    summary = run_country_imputation(db_path, auto_write=True)

Usage (CLI): openalex impute-country --db-path data/db/mydata.duckdb
"""

import duckdb
import pandas as pd
import pycountry
import requests
import time
import re

# ---------------------------------------------------------------------------
# CONFIG — adjust per dataset
# ---------------------------------------------------------------------------
TABLE = "contributions"              # table holding country_code + raw text
COUNTRY_COL = "country_code"
INSTITUTION_COL = "institution_id"
RAW_TEXT_COL = "raw_affiliation_string"
ROW_ID_COL = "row_id"
INSTITUTIONS_TABLE = "institutions"  # table with institution_id -> country_code
INSTITUTIONS_ID_COL = "id"           # institutions table's own primary key
INSTITUTIONS_ROR_COL = "ror_id"      # institutions table's ROR ID column

# ROR live API config (Pass 0)
ROR_API_URL = "https://api.ror.org/v2/organizations"
ROR_MAX_RETRIES = 3
ROR_REQUEST_TIMEOUT = 10
ROR_REQUEST_DELAY = 0.5  # seconds between calls — be polite to the API
ROR_REPORT = "ror_country_lookup_report.csv"

# Optional: extend with other countries' states/provinces as new datasets
# reveal them to be common in the raw text (e.g. Chinese provinces, German
# Länder, Canadian provinces).
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]
# NOTE: "Georgia" is ambiguous (US state vs. country). If country-name
# matching (Pass 2) already ran, exclude "Georgia" here to avoid conflicts,
# or add a disambiguation rule (e.g. check for a US zip code pattern nearby).

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry",
    "Chandigarh",
]

REVIEW_SAMPLE_SIZE = 15  # rows to manually eyeball per pass before writing


def connect(db_path):
    return duckdb.connect(db_path)


def null_country_count(con):
    return con.sql(
        f"SELECT COUNT(*) FROM {TABLE} WHERE {COUNTRY_COL} IS NULL"
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# PASS 0 — ROR API institution repair (fixes institutions.country_code)
# ---------------------------------------------------------------------------
def is_missing_sql(col):
    return f"({col} IS NULL OR TRIM({col}) = '')"


def extract_ror_id(raw_ror):
    """Pulls the bare 9-character ROR ID from a value that may be a full
    URL (https://ror.org/04ttjf776) or already bare (04ttjf776)."""
    if not isinstance(raw_ror, str):
        return None
    match = re.search(r'([a-z0-9]{9})$', raw_ror.strip().lower())
    return match.group(1) if match else None


def run_pass_0_ror_repair(con, test_limit=None):
    """Repairs institutions.country_code by looking up each institution's
    ror_id against the LIVE ROR API. This is an exact-ID lookup, not a
    guess, so it auto-writes without the manual-review step the text-based
    passes require. Validated at ~99% hit rate (102/103) on a real test.
    Run this FIRST, every time — it's what makes Pass A actually productive
    instead of a no-op."""
    missing_count = con.execute(
        f"SELECT COUNT(*) FROM {INSTITUTIONS_TABLE} WHERE {is_missing_sql(COUNTRY_COL)}"
    ).fetchone()[0]
    print(f"[Pass 0] Missing institution countries: {missing_count:,}")
    if missing_count == 0:
        print("[Pass 0] No missing institution countries.")
        return 0

    df = con.execute(f"""
        SELECT {INSTITUTIONS_ID_COL} AS institution_id, {INSTITUTIONS_ROR_COL} AS ror_id
        FROM {INSTITUTIONS_TABLE}
        WHERE {is_missing_sql(COUNTRY_COL)}
    """).fetchdf()

    df["clean_ror"] = df["ror_id"].apply(extract_ror_id)
    valid_rors = df["clean_ror"].dropna().drop_duplicates().tolist()
    print(f"[Pass 0] Rows without valid ROR ID: {df['clean_ror'].isna().sum():,}")
    print(f"[Pass 0] Unique ROR IDs: {len(valid_rors):,}")
    if not valid_rors:
        print("[Pass 0] No valid ROR IDs found.")
        return 0

    if test_limit is not None:
        print(f"[Pass 0][TEST MODE] Capping to first {test_limit} ROR IDs.")
        valid_rors = valid_rors[:test_limit]

    session = requests.Session()
    results = []
    for index, ror_id in enumerate(valid_rors, start=1):
        print(f"[Pass 0][{index}/{len(valid_rors)}] {ror_id}")
        url = f"{ROR_API_URL}/{ror_id}"
        country = None
        for attempt in range(ROR_MAX_RETRIES):
            try:
                response = session.get(url, timeout=ROR_REQUEST_TIMEOUT)
                if response.status_code == 404:
                    break
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                for location in data.get("locations", []):
                    details = location.get("geonames_details", {})
                    country = details.get("country_code")
                    if country:
                        country = str(country).upper()
                        break
                break
            except requests.RequestException as exc:
                if attempt == ROR_MAX_RETRIES - 1:
                    print(f"  [Pass 0] ROR request failed: {exc}")
                    break
                time.sleep(2 ** attempt)
        if country:
            results.append({"ror_id": ror_id, "country_code": country})
        time.sleep(ROR_REQUEST_DELAY)

    lookup = pd.DataFrame(results, columns=["ror_id", "country_code"])
    lookup.to_csv(ROR_REPORT, index=False)
    if lookup.empty:
        print("[Pass 0] No ROR countries found.")
        return 0

    con.register("ror_lookup", lookup)
    con.execute(f"""
        UPDATE {INSTITUTIONS_TABLE} AS i
        SET {COUNTRY_COL} = r.country_code
        FROM ror_lookup AS r
        WHERE LOWER(REGEXP_REPLACE(CAST(i.{INSTITUTIONS_ROR_COL} AS VARCHAR), '^https?://ror\\.org/', '')) = LOWER(r.ror_id)
          AND {is_missing_sql(f"i.{COUNTRY_COL}")}
    """)
    con.commit()
    con.unregister("ror_lookup")
    print(f"[Pass 0] institutions.country_code repaired for {len(lookup):,} institutions.")
    return len(lookup)


# ---------------------------------------------------------------------------
# PASS A — join-fill from a matched institution's own country_code
# ---------------------------------------------------------------------------
def run_pass_a(con):
    """Fills country_code where institution_id is known and the linked
    institution has a non-null country_code. In the original run this
    recovered 0 rows (linked institutions were also null) but it's a
    free check worth running first on every new dataset."""
    before = null_country_count(con)
    con.sql(f"""
        UPDATE {TABLE}
        SET {COUNTRY_COL} = (
            SELECT i.{COUNTRY_COL}
            FROM {INSTITUTIONS_TABLE} i
            WHERE i.id = {TABLE}.{INSTITUTION_COL}
        )
        WHERE {TABLE}.{COUNTRY_COL} IS NULL
          AND {TABLE}.{INSTITUTION_COL} IS NOT NULL
    """)
    after = null_country_count(con)
    print(f"[Pass A] Recovered {before - after} rows via institution join.")
    return before - after


# ---------------------------------------------------------------------------
# PASS B — country-name keyword match via pycountry
# ---------------------------------------------------------------------------
def build_country_lookup():
    country_list = [(c.name, c.alpha_2) for c in pycountry.countries]
    df = pd.DataFrame(country_list, columns=["country_name", "country_code"])
    # longest-first avoids "Guinea" matching inside "Equatorial Guinea"
    df = df.sort_values(by="country_name", key=lambda x: x.str.len(),
                         ascending=False)
    return df


def find_country_code(text, country_lookup):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    for _, row in country_lookup.iterrows():
        if row["country_name"].lower() in text_lower:
            return row["country_code"]
    return None


def run_pass_b(con, country_lookup, auto_write=False):
    targets = con.sql(f"""
        SELECT {ROW_ID_COL}, {RAW_TEXT_COL}
        FROM {TABLE}
        WHERE {COUNTRY_COL} IS NULL AND {RAW_TEXT_COL} IS NOT NULL
    """).df()

    targets["matched_country_code"] = targets[RAW_TEXT_COL].apply(
        lambda x: find_country_code(x, country_lookup)
    )
    matched = targets[targets["matched_country_code"].notnull()]
    print(f"[Pass B] {len(matched)} candidate matches out of {len(targets)} targets.")

    if len(matched) == 0:
        return 0

    # ALWAYS spot-check before writing — do not skip this step on a new dataset.
    sample = matched.sample(min(REVIEW_SAMPLE_SIZE, len(matched)), random_state=1)
    print("\n--- SPOT CHECK SAMPLE (review before writing) ---")
    for _, row in sample.iterrows():
        print(row["matched_country_code"], "|", row[RAW_TEXT_COL])
    print("--- END SAMPLE ---\n")

    if not auto_write:
        print("Review the sample above. Re-run with auto_write=True to commit.")
        return len(matched)

    con.register("pass_b_matches", matched)
    con.sql(f"""
        UPDATE {TABLE}
        SET {COUNTRY_COL} = t.matched_country_code
        FROM pass_b_matches t
        WHERE {TABLE}.{ROW_ID_COL} = t.{ROW_ID_COL}
          AND t.matched_country_code IS NOT NULL
    """)
    con.unregister("pass_b_matches")
    print(f"[Pass B] Wrote {len(matched)} rows.")
    return len(matched)


# ---------------------------------------------------------------------------
# PASS C — sub-national region matching (US states, Indian states, ...)
# ---------------------------------------------------------------------------
def run_region_pass(con, region_list, region_country_code, label,
                     exclude_terms=None, auto_write=False):
    """Generic region-name pass. Pass a region_list (e.g. US_STATES) and
    the country code it maps to. exclude_terms lets you skip ambiguous
    names (e.g. "Georgia") that could collide with a country name already
    matched in Pass B."""
    exclude_terms = set(t.lower() for t in (exclude_terms or []))
    active_regions = [r for r in region_list if r.lower() not in exclude_terms]

    targets = con.sql(f"""
        SELECT {ROW_ID_COL}, {RAW_TEXT_COL}
        FROM {TABLE}
        WHERE {COUNTRY_COL} IS NULL AND {RAW_TEXT_COL} IS NOT NULL
    """).df()

    def find_region(text):
        if not isinstance(text, str):
            return None
        text_lower = text.lower()
        for region in active_regions:
            if region.lower() in text_lower:
                return region
        return None

    targets["matched_region"] = targets[RAW_TEXT_COL].apply(find_region)
    matched = targets[targets["matched_region"].notnull()]
    print(f"[{label}] {len(matched)} candidate matches out of {len(targets)} targets.")

    if len(matched) == 0:
        return 0

    print(matched["matched_region"].value_counts())  # check for skew/ambiguity

    # ALWAYS spot-check actual raw text too, not just the aggregate counts —
    # a clean-looking value_counts breakdown can still hide bad individual
    # matches (e.g. a region name matching inside an unrelated word).
    sample = matched.sample(min(REVIEW_SAMPLE_SIZE, len(matched)), random_state=1)
    print(f"\n--- SPOT CHECK SAMPLE ({label}, review before writing) ---")
    for _, row in sample.iterrows():
        print(row["matched_region"], "|", row[RAW_TEXT_COL])
    print("--- END SAMPLE ---\n")

    if not auto_write:
        print("Review the breakdown and sample above. Re-run with auto_write=True to commit.")
        return len(matched)

    con.register(f"{label}_matches", matched)
    con.sql(f"""
        UPDATE {TABLE}
        SET {COUNTRY_COL} = '{region_country_code}'
        FROM {label}_matches t
        WHERE {TABLE}.{ROW_ID_COL} = t.{ROW_ID_COL}
    """)
    con.unregister(f"{label}_matches")
    print(f"[{label}] Wrote {len(matched)} rows.")
    return len(matched)


# ---------------------------------------------------------------------------
# ORCHESTRATOR — called by the CLI command (openalex impute-country)
# ---------------------------------------------------------------------------
def run_country_imputation(db_path, auto_write=False, ror_test_limit=None):
    """Runs the full country-imputation pipeline (Pass 0 through Pass C)
    against the given .duckdb file.

    auto_write: if False (default), Pass B and the region passes only find
        and print candidate matches for review — nothing is written for
        those passes. Pass 0 and Pass A always write, since Pass 0 is an
        exact-ID lookup (no ambiguity) and Pass A is a straight join on
        already-confirmed institution data.
    ror_test_limit: caps Pass 0 to the first N ROR IDs — useful for a quick
        validation run on a brand-new dataset before committing to the
        full (potentially long) live-API batch.

    Returns a dict summary of what was recovered at each pass.
    """
    con = connect(db_path)
    start_nulls = null_country_count(con)
    print(f"Starting null {COUNTRY_COL}: {start_nulls}\n")

    ror_recovered = run_pass_0_ror_repair(con, test_limit=ror_test_limit)
    pass_a_recovered = run_pass_a(con)

    country_lookup = build_country_lookup()
    pass_b_recovered = run_pass_b(con, country_lookup, auto_write=auto_write)

    us_recovered = run_region_pass(con, US_STATES, "US", label="us_state_pass",
                                    exclude_terms=["Georgia"], auto_write=auto_write)
    in_recovered = run_region_pass(con, INDIAN_STATES, "IN", label="indian_state_pass",
                                    auto_write=auto_write)

    end_nulls = null_country_count(con)
    con.close()

    print(f"\nFinal null {COUNTRY_COL}: {end_nulls}")
    print(f"Total recovered this run: {start_nulls - end_nulls}")

    return {
        "start_nulls": start_nulls,
        "end_nulls": end_nulls,
        "total_recovered": start_nulls - end_nulls,
        "ror_institutions_repaired": ror_recovered,
        "pass_a_recovered": pass_a_recovered,
        "pass_b_candidates": pass_b_recovered,
        "us_state_candidates": us_recovered,
        "indian_state_candidates": in_recovered,
        "auto_write": auto_write,
    }