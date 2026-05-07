#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path



MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = ""  # Paste your Groq API key here
CSV_PATH = Path(
    "/Users/sruthikalyani/.copilot/session-state/3ab2dd19-244c-45aa-b52b-09e18f0ff5d2/files/impute_compare_20.csv"
)


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
    payload_str = json.dumps(body)
    last_error = ""
    t0 = time.time()
    for attempt in range(3):
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "https://api.groq.com/openai/v1/chat/completions",
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "-d",
                payload_str,
            ],
            capture_output=True,
            text=True,
        )

        raw = (proc.stdout or "").strip()
        if proc.returncode != 0:
            last_error = (proc.stderr or "curl failed").strip()
            time.sleep(1.5 * (attempt + 1))
            continue

        try:
            payload = json.loads(raw)
        except Exception:
            last_error = f"non-json API response: {raw[:300]}"
            time.sleep(1.5 * (attempt + 1))
            continue

        if "error" in payload:
            err = payload["error"]
            msg = err.get("message", str(err))
            code = err.get("code", "unknown")
            raise RuntimeError(f"Groq API error ({code}): {msg}")

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
    api_key = _extract_api_key(GROQ_API_KEY)
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY inside compare_impute_groq.py (must contain a gsk_... token).")

    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"CSV has no rows: {CSV_PATH}")

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

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("rows:", len(rows))
    print("agreement_count:", agreement_count)
    print("agreement_rate:", round(agreement_count / len(rows), 3))
    print("llm_avg_seconds:", round(total_latency / len(rows), 3))
    print("llm_total_seconds:", round(total_latency, 3))
    print("updated_csv:", CSV_PATH)


if __name__ == "__main__":
    main()
