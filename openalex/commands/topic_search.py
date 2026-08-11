"""Analyze OpenAlex topics for papers with missing DOIs/topic coverage."""

from __future__ import annotations

import csv
import re
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import click
import requests


def importance(freq: int, total: int) -> str:
    """Return importance rating based on topic coverage."""
    pct = freq / total * 100

    if pct >= 20:
        return "★★★★★ Critical"
    elif pct >= 10:
        return "★★★★ High"
    elif pct >= 5:
        return "★★★ Medium"
    elif pct >= 2:
        return "★★ Low"
    elif pct >= 1:
        return "★ Very Low"
    else:
        return "✩ Negligible"


@click.command("topic-search")
@click.argument(
    "doi_file",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    default="config/missing_dois.txt",
)
@click.option(
    "--topics-file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    default="config/topics.txt",
    show_default=True,
    help="File containing the currently selected OpenAlex topic IDs.",
)
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    default="missing_topic_analysis.csv",
    show_default=True,
    help="Output CSV file.",
)
@click.option(
    "--delay",
    type=float,
    default=0.12,
    show_default=True,
    help="Delay between OpenAlex API requests.",
)
def topic_search_command(
    doi_file: Path,
    topics_file: Path,
    output: Path,
    delay: float,
) -> None:
    """Analyze OpenAlex topics for papers in DOI_FILE."""

    # ============================================================
    # Create DOI file if it doesn't exist
    # ============================================================

    doi_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not doi_file.exists():
        doi_file.touch()

    # ============================================================
    # Load current topics
    # ============================================================

    current_topics = {
        line.strip()
        for line in topics_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    # ============================================================
    # Read DOIs
    # ============================================================

    dois: list[str] = []

    with doi_file.open(encoding="utf-8") as f:
        for line in f:
            match = re.search(r"10\.\S+", line)

            if match:
                doi = match.group(0).rstrip(
                    ".,;)]}>"
                )
                dois.append(doi)

    # Remove duplicates
    dois = list(dict.fromkeys(dois))

    if not dois:
        raise click.ClickException(
            f"No DOIs found in {doi_file}"
        )

    # ============================================================
    # Data structures
    # ============================================================

    counter: Counter[str] = Counter()

    topic_info: dict[
        str,
        dict[str, str],
    ] = {}

    topic_dois: dict[
        str,
        list[str],
    ] = {}

    # ============================================================
    # Query OpenAlex
    # ============================================================

    for doi in dois:

        encoded = urllib.parse.quote(
            doi,
            safe="",
        )

        url = (
            "https://api.openalex.org/works/"
            f"https://doi.org/{encoded}"
        )

        try:
            response = requests.get(
                url,
                timeout=30,
            )

            if response.status_code != 200:
                continue

            work = response.json()

            topics = work.get(
                "topics",
                [],
            )

            for topic in topics:

                topic_id = topic["id"].split("/")[-1]

                counter[topic_id] += 1

                topic_info[topic_id] = {
                    "topic": topic.get(
                        "display_name",
                        "",
                    ),
                    "subfield": topic.get(
                        "subfield",
                        {},
                    ).get(
                        "display_name",
                        "",
                    ),
                    "field": topic.get(
                        "field",
                        {},
                    ).get(
                        "display_name",
                        "",
                    ),
                    "domain": topic.get(
                        "domain",
                        {},
                    ).get(
                        "display_name",
                        "",
                    ),
                }

                topic_dois.setdefault(
                    topic_id,
                    [],
                ).append(doi)

        except requests.RequestException:
            continue

        except Exception:
            continue

        time.sleep(delay)

    # ============================================================
    # Build CSV rows
    # ============================================================

    rows = []

    for topic_id, frequency in counter.most_common():

        info = topic_info[topic_id]

        coverage = (
            frequency / len(dois) * 100
        )

        status = (
            "Already Present"
            if topic_id in current_topics
            else "NEW"
        )

        score = importance(
            frequency,
            len(dois),
        )

        rows.append(
            [
                topic_id,
                info["topic"],
                info["subfield"],
                info["field"],
                info["domain"],
                frequency,
                round(coverage, 2),
                score,
                status,
                topic_dois[topic_id][0],
            ]
        )

    # ============================================================
    # Save CSV
    # ============================================================

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Topic ID",
                "Topic",
                "Subfield",
                "Field",
                "Domain",
                "Frequency",
                "Coverage %",
                "Importance",
                "Status",
                "Example DOI",
            ]
        )

        writer.writerows(rows)

    click.echo(f"Saved: {output}")