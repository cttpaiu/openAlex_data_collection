#!/usr/bin/env python3
"""Eval script: compare rule-based vs Groq LLM country imputation on a sample CSV.

Usage:
    GROQ_API_KEY=gsk_... python compare_impute_groq.py --csv path/to/sample.csv
    python compare_impute_groq.py --csv path/to/sample.csv --groq-api-key gsk_...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from pathlib import Path

MODEL = "llama-3.1-8b-instant"


def ask_llm(api_key: str, affiliation: str) -> tuple[str, str, float]:
    prompt = (
        "Infer ISO-3166-1 alpha-2 country code from this affiliation string.\n"
        "Return strict JSON only in this format: "
        '{"country_code":"XX"|null,"confidence":0-1,"reason":"..."}\n'
        f"Affiliation: {affiliation}"
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 120,
    }
    body_bytes = json.dumps(body).encode()
    last_error = ""
    t0 = time.time()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body_bytes,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
            continue

        if "error" in payload:
            err = payload["error"]
            raise RuntimeError(f"Groq API error ({err.get('code','unknown')}): {err.get('message', err)}")

        latency = time.time() - t0
        content = payload["choices"][0]["message"]["content"].strip()
        try:
            parsed = json.loads(content)
            return (parsed.get("country_code") or "", parsed.get("reason", ""), latency)
        except Exception:
            return ("", f"non-json response: {content[:200]}", latency)

    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def _extract_api_key(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    match = re.search(r"(gsk_[A-Za-z0-9]+)", value)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule vs LLM country imputation")
    parser.add_argument("--csv", required=True, help="Path to input CSV with raw_affiliation_string and rule_country columns")
    parser.add_argument("--groq-api-key", default=None, help="Groq API key (or set GROQ_API_KEY env var)")
    args = parser.parse_args()

    raw_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    api_key = _extract_api_key(raw_key)
    if not api_key:
        raise SystemExit("Provide Groq API key via --groq-api-key or GROQ_API_KEY env var (must contain gsk_... token).")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_path}")

    total_latency = 0.0
    agreement_count = 0

    for row in rows:
        llm_country, reason, latency = ask_llm(api_key, row["raw_affiliation_string"])
        row["llm_country"] = llm_country
        row["llm_reason"] = reason
        row["agree_rule_vs_llm"] = "yes" if llm_country and llm_country == row["rule_country"] else "no"
        if row["agree_rule_vs_llm"] == "yes":
            agreement_count += 1
        total_latency += latency

    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("rows:", len(rows))
    print("agreement_count:", agreement_count)
    print("agreement_rate:", round(agreement_count / len(rows), 3))
    print("llm_avg_seconds:", round(total_latency / len(rows), 3))
    print("llm_total_seconds:", round(total_latency, 3))
    print("updated_csv:", csv_path)


if __name__ == "__main__":
    main()
