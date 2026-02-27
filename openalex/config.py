"""Config loader and template generator for openalex CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

CONFIG_DIR = Path("config")
DATA_DIR = Path("data")

DEFAULT_CONFIG: dict[str, Any] = {
    "api": {
        "key": "YOUR_API_KEY_HERE",
        "email": "your@email.com",
        "base_url": "https://api.openalex.org/works",
    },
    "filters": {
        "date_from": "2003-01-01",
        "date_to": "2024-12-31",
        "doc_types": ["article", "review", "proceedings-article"],
    },
    "keywords_file": "config/keywords.txt",
    "topics_file": "config/topics.txt",
    "collection": {
        "batch_size_topics": 10,
        "per_page": 200,
        "concurrent_requests": 10,
        "max_retries": 5,
        "retry_delay": 2,
    },
    "output": {
        "jsonl_dir": "data/raw/",
        "db_dir": "data/db/",
    },
}

DEFAULT_KEYWORDS = '''\
(
("quantum computing" OR "quantum computation" OR "quantum processor" OR
"quantum information processing" OR "quantum information science" OR
"quantum supremacy" OR "quantum advantage" OR "quantum speedup" OR
"measurement based quantum computing" OR "MBQC" OR "blind quantum computing") OR

("quantum algorithm" OR "quantum circuit" OR "quantum gate" OR
"quantum complexity" OR "quantum error correction" OR
"fault-tolerant quantum computing" OR "quantum fault tolerance" OR
"quantum simulation" OR "quantum simulator" OR
"variational quantum" OR "parameterized quantum circuit") OR

("qubit" OR "qutrit" OR "qudit") OR

("transmon" OR "fluxonium" OR "trapped ion" OR "ion trap" OR
"boson sampling" OR "majorana zero mode" OR
"quantum annealing" OR "quantum annealer" OR "adiabatic quantum computing" OR
(("neutral atom" OR "rydberg atom") AND ("qubit" OR "gate" OR "array" OR "simulator" OR "entanglement" OR "computer")) OR
("quantum dot" AND ("qubit" OR "quantum gate" OR "fidelity" OR "Rabi oscillation")) OR
"hybrid quantum system") OR

("surface code" OR "stabilizer code" OR "quantum decoherence" OR "error mitigation" OR
"quantum noise model" OR "dynamical decoupling") OR

("quantum software" OR "quantum programming" OR "quantum compiler" OR
"QPU" OR "quantum cloud" OR "distributed quantum computing" OR
"hybrid classical-quantum" OR "quantum internet" OR "quantum network" OR
"quantum repeater" OR "quantum memory") OR

("Shor algorithm" OR "Grover algorithm" OR "Simon algorithm" OR
"quantum Fourier transform" OR "quantum phase estimation" OR
"Deutsch-Jozsa" OR "Bernstein-Vazirani" OR "amplitude amplification" OR
"variational quantum eigensolver" OR "VQE" OR
"quantum approximate optimization algorithm" OR "QAOA" OR
"quantum machine learning" OR "quantum neural network" OR "QNN" OR "quantum kernel method")
)
'''

DEFAULT_TOPICS_HEADER = """\
# One OpenAlex topic ID per line (format: T + 5 digits, e.g. T10020)
# Run: openalex get-topics   to discover topic IDs from your keyword search
# Example IDs for quantum computing:
# T10020
# T10682
"""


@dataclass
class AppConfig:
    api_key: str
    email: str
    base_url: str
    date_from: str
    date_to: str
    doc_types: list[str]
    keywords_file: str
    topics_file: str
    batch_size_topics: int
    per_page: int
    concurrent_requests: int
    max_retries: int
    retry_delay: int
    jsonl_dir: str
    db_dir: str

    _keywords: str | None = field(default=None, repr=False)
    _topics: list[str] | None = field(default=None, repr=False)

    def get_keywords(self) -> str:
        if self._keywords is None:
            kw_path = Path(self.keywords_file)
            if not kw_path.exists():
                raise FileNotFoundError(
                    f"Keywords file not found: {self.keywords_file}\n"
                    "Run 'openalex init' to create a template."
                )
            raw = kw_path.read_text(encoding="utf-8").strip()
            # Collapse all whitespace (newlines, tabs, multiple spaces) to a
            # single space so the query is safe to embed in an API filter string.
            import re
            self._keywords = re.sub(r"\s+", " ", raw)
        return self._keywords

    def get_topics(self) -> list[str]:
        if self._topics is None:
            topics_path = Path(self.topics_file)
            if not topics_path.exists():
                return []
            lines = topics_path.read_text(encoding="utf-8").splitlines()
            self._topics = [
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
        return self._topics

    def validate_api_key(self) -> None:
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            console.print(
                "[bold red]✗ API key not configured.[/bold red]\n"
                "Edit [cyan]config/collection.yml[/cyan] and set [cyan]api.key[/cyan].",
            )
            raise SystemExit(1)


def load_config(config_path: str = "config/collection.yml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        console.print(
            f"[bold red]✗ Config file not found:[/bold red] {config_path}\n"
            "Run [cyan]openalex init[/cyan] to create a template.",
        )
        raise SystemExit(1)

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    api = raw.get("api", {})
    filters = raw.get("filters", {})
    coll = raw.get("collection", {})
    out = raw.get("output", {})

    return AppConfig(
        api_key=api.get("key", ""),
        email=api.get("email", ""),
        base_url=api.get("base_url", "https://api.openalex.org/works"),
        date_from=filters.get("date_from", "2003-01-01"),
        date_to=filters.get("date_to", "2024-12-31"),
        doc_types=filters.get("doc_types", ["article", "review", "proceedings-article"]),
        keywords_file=raw.get("keywords_file", "config/keywords.txt"),
        topics_file=raw.get("topics_file", "config/topics.txt"),
        batch_size_topics=coll.get("batch_size_topics", 10),
        per_page=coll.get("per_page", 200),
        concurrent_requests=coll.get("concurrent_requests", 10),
        max_retries=coll.get("max_retries", 5),
        retry_delay=coll.get("retry_delay", 2),
        jsonl_dir=out.get("jsonl_dir", "data/raw/"),
        db_dir=out.get("db_dir", "data/db/"),
    )


@click.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing config files")
def init_command(force: bool) -> None:
    """Create config template files to get started."""
    created = []
    skipped = []

    dirs = [CONFIG_DIR, DATA_DIR / "raw", DATA_DIR / "db"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    config_file = CONFIG_DIR / "collection.yml"
    if not config_file.exists() or force:
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        created.append(str(config_file))
    else:
        skipped.append(str(config_file))

    keywords_file = CONFIG_DIR / "keywords.txt"
    if not keywords_file.exists() or force:
        keywords_file.write_text(DEFAULT_KEYWORDS, encoding="utf-8")
        created.append(str(keywords_file))
    else:
        skipped.append(str(keywords_file))

    topics_file = CONFIG_DIR / "topics.txt"
    if not topics_file.exists() or force:
        topics_file.write_text(DEFAULT_TOPICS_HEADER, encoding="utf-8")
        created.append(str(topics_file))
    else:
        skipped.append(str(topics_file))

    lines = []
    for f in created:
        lines.append(f"  [green]✓[/green] Created  [cyan]{f}[/cyan]")
    for f in skipped:
        lines.append(f"  [yellow]–[/yellow] Skipped  [cyan]{f}[/cyan] (already exists, use --force to overwrite)")

    console.print(
        Panel(
            "\n".join(lines) + "\n\n[bold]Next step:[/bold] Edit [cyan]config/collection.yml[/cyan] and set your API key.",
            title="[bold green]openalex init[/bold green]",
            expand=False,
        )
    )
