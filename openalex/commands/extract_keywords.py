"""Command: extract-keywords - TF-IDF over title+abstract, optionally re-scored
with KeyBERT, store ranked keywords."""

from __future__ import annotations

import json
from pathlib import Path

import click
import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()

_MIN_DOC_CHARS = 20
_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"


_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls (CFB/OLE)
_ZIP_SIGNATURE = b"PK\x03\x04"  # .xlsx / .docx / any modern Office zip


def _load_dataframe(path: Path) -> pl.DataFrame:
    """Load a DataFrame, detecting the real file format from its content
    rather than trusting the extension.

    Export tools (Web of Science, Scopus, etc.) routinely hand out files
    named '.xls' that are actually HTML tables or plain delimited text
    (CSV/TSV) under the hood. Trusting the extension causes cryptic binary
    parser errors like "Invalid OLE signature" on files that were never
    real Excel binaries. This reads the first bytes to pick the correct
    reader no matter what the file is named -- so .csv, .tsv, .txt, .xls,
    .xlsx, and mislabeled HTML-as-.xls exports all just work.
    """
    header = path.read_bytes()[:8]

    if header.startswith(_ZIP_SIGNATURE) or header.startswith(_OLE_SIGNATURE):
        return pl.read_excel(path)

    text_sample = _read_text_sample(path)
    if text_sample.lstrip()[:1] == "<" and "<table" in text_sample.lower():
        return _load_html_table(path)

    return _load_delimited_text(path)


def _read_text_sample(path: Path, n_bytes: int = 4096) -> str:
    raw = path.read_bytes()[:n_bytes]
    for enc in ("utf-8", "utf-8-sig", "windows-1252", "iso8859-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _load_html_table(path: Path) -> pl.DataFrame:
    try:
        import pandas as pd
    except ImportError as exc:
        raise click.ClickException(
            "This file is HTML-formatted (common for Web of Science / Scopus "
            "exports saved with an .xls extension). Reading it requires "
            "pandas + lxml: install with `uv add pandas lxml`."
        ) from exc

    tables = pd.read_html(path)
    if not tables:
        raise click.ClickException(f"No <table> found in HTML content: {path}")
    # Some export pages wrap the real data table alongside small nav/header
    # tables -- take the largest one by cell count.
    biggest = max(tables, key=lambda df: df.shape[0] * df.shape[1])
    return pl.from_pandas(biggest)


def _load_delimited_text(path: Path) -> pl.DataFrame:
    import csv as csv_module

    sample = _read_text_sample(path)
    try:
        sep = csv_module.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv_module.Error:
        sep = ","  # can't detect -- default to comma

    encodings = ["utf8", "utf8-lossy", "windows-1252", "iso8859-1"]
    for enc in encodings:
        try:
            return pl.read_csv(
                path,
                encoding=enc,
                separator=sep,
                null_values="n/a",
                infer_schema_length=10000,
            )
        except Exception:
            continue

    raise click.ClickException(
        f"Could not read {path} as delimited text with any supported encoding."
    )


def _build_documents(df: pl.DataFrame, title_col: str, abstract_col: str) -> list[str]:
    missing = [c for c in (title_col, abstract_col) if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"Column(s) not found: {missing}. Available columns: {df.columns}"
        )
    docs: list[str] = []
    for row in df.iter_rows(named=True):
        title = (row.get(title_col) or "").strip()
        abstract = (row.get(abstract_col) or "").strip()
        combined = f"{title}. {abstract}".strip(". ").strip()
        if len(combined) >= _MIN_DOC_CHARS:
            docs.append(combined)
    return docs


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

custom_stopwords = list(ENGLISH_STOP_WORDS.union({
    "english", "rights", "reserved", "elsevier rights", "elsevier rights reserved",
    "published elsevier", "paper presents", "published elsevier rights", "elsevier",
    "published", "paper", "presents", "comprises", "following", "methord", "based",
    "device", "present", "invention", "methord steps", "steps", "classification",
    "classification method", "method", "methods", "methodology", "methodologies",
    "methodological", "methodological approach", "approach", "approaches",
    "approach method", "approach methods", "using methord", "using methods",
    "using approach", "using approaches", "using methodology",
    "using methodologies", "using methodological approach", "storage medium",
    "storage media", "storage medium comprising", "storage media comprising",
    "medium", "media", "comprising", "comprises", "comprise", "method detecting",
    "data set", "method deep",
}))


def _score_terms(
    docs: list[str], ngram_min: int, ngram_max: int, min_df: int, max_df: float
) -> list[tuple[str, float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max),
        stop_words=custom_stopwords,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
    )
    matrix = vectorizer.fit_transform(docs)
    mean_scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    pairs = list(zip(terms, mean_scores))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


def _keybert_score(
    docs: list[str],
    candidates: list[str],
    model_name: str,
) -> dict[str, float]:
    """Re-score TF-IDF candidate terms by semantic relevance to each document.

    This computes the same thing KeyBERT does under the hood (cosine
    similarity between a document embedding and candidate-phrase embeddings)
    but does it directly via sentence-transformers, bypassing KeyBERT's
    internal candidate CountVectorizer. That internal vectorizer uses its own
    tokenizer to reconstruct candidate phrases from the document text, which
    can silently fail to match multi-word / hyphenated n-grams generated by a
    *different* tokenizer (ours) -- resulting in every score coming back 0
    with no visible error. Going direct avoids that class of failure and is
    also faster: candidates and docs are each embedded once in a batch,
    rather than reloading a restricted vectorizer per document.

    A term's score is averaged across every document whose text contains it
    (case-insensitive substring match). Terms that never appear score 0.
    """
    from sentence_transformers import SentenceTransformer, util

    console.print(f"[dim]Loading embedding model '{model_name}'...[/dim]")
    model = SentenceTransformer(model_name)

    candidate_lower = [c.lower() for c in candidates]
    term_scores: dict[str, list[float]] = {t: [] for t in candidates}

    with console.status("[dim]Embedding candidate terms...[/dim]"):
        candidate_embeddings = model.encode(
            candidates, show_progress_bar=False, convert_to_tensor=True
        )

    with console.status(f"[dim]Embedding {len(docs)} documents...[/dim]"):
        doc_embeddings = model.encode(
            docs, show_progress_bar=False, convert_to_tensor=True, batch_size=32
        )

    matched_any = False
    for doc, doc_emb in zip(docs, doc_embeddings):
        doc_lower = doc.lower()
        present_idx = [i for i, c in enumerate(candidate_lower) if c in doc_lower]
        if not present_idx:
            continue
        matched_any = True
        sims = util.cos_sim(doc_emb, candidate_embeddings[present_idx])[0].tolist()
        for idx, sim in zip(present_idx, sims):
            term_scores[candidates[idx]].append(sim)

    if not matched_any:
        console.print(
            "[yellow]! No candidate terms matched any document text -- "
            "check that --candidate-pool terms and doc text use the same casing/format.[/yellow]"
        )

    return {t: (sum(s) / len(s) if s else 0.0) for t, s in term_scores.items()}


def _blend_scores(
    tfidf_pairs: list[tuple[str, float]],
    keybert_scores: dict[str, float],
    alpha: float,
) -> list[tuple[str, float, float, float]]:
    """Combine min-max normalized TF-IDF and KeyBERT scores.

    Returns rows of (term, raw_tfidf, keybert_score, combined_score),
    sorted by combined_score descending. alpha=1.0 is pure TF-IDF,
    alpha=0.0 is pure KeyBERT.
    """
    tfidf_vals = [s for _, s in tfidf_pairs]
    lo, hi = min(tfidf_vals), max(tfidf_vals)
    span = (hi - lo) or 1.0
    tfidf_norm = [(s - lo) / span for s in tfidf_vals]

    kb_vals = [keybert_scores.get(t, 0.0) for t, _ in tfidf_pairs]
    klo, khi = min(kb_vals), max(kb_vals)
    kspan = (khi - klo) or 1.0
    kb_norm = [(s - klo) / kspan for s in kb_vals]

    rows = []
    for i, (term, raw_tfidf) in enumerate(tfidf_pairs):
        combined = alpha * tfidf_norm[i] + (1 - alpha) * kb_norm[i]
        rows.append((term, float(raw_tfidf), float(kb_norm[i]), float(combined)))

    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


def _print_table(top: list[tuple[str, float]]) -> None:
    table = Table(title=f"Top {len(top)} TF-IDF keywords", show_lines=False)
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Keyword", style="white")
    table.add_column("Mean TF-IDF", justify="right", style="green")
    for i, (term, score) in enumerate(top, start=1):
        table.add_row(str(i), term, f"{score:.4f}")
    console.print(table)


def _print_blended_table(rows: list[tuple[str, float, float, float]]) -> None:
    table = Table(title=f"Top {len(rows)} keywords (TF-IDF + KeyBERT)", show_lines=False)
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Keyword", style="white")
    table.add_column("TF-IDF", justify="right", style="green")
    table.add_column("KeyBERT", justify="right", style="magenta")
    table.add_column("Combined", justify="right", style="bold yellow")
    for i, (term, tfidf_s, kb_s, combined) in enumerate(rows, start=1):
        table.add_row(str(i), term, f"{tfidf_s:.4f}", f"{kb_s:.4f}", f"{combined:.4f}")
    console.print(table)


def _confirm_overwrite(path: Path, force: bool) -> bool:
    if not path.exists() or force:
        return True
    import questionary

    return bool(questionary.confirm(f"{path} exists. Overwrite?", default=False).ask())


def _write_output(
    pairs: list[tuple],
    output_path: Path,
    fmt: str,
    source: Path,
    params: dict,
    blended: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if blended:
        keywords = [
            {"term": t, "tfidf_score": tf, "keybert_score": kb, "combined_score": c}
            for t, tf, kb, c in pairs
        ]
    else:
        keywords = [{"term": t, "score": float(s)} for t, s in pairs]

    if fmt == "json":
        payload = {"_meta": {"source": str(source), "params": params}, "keywords": keywords}
        output_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    elif fmt == "csv":
        import csv

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if blended:
                writer.writerow(["rank", "term", "tfidf_score", "keybert_score", "combined_score"])
                for i, (t, tf, kb, c) in enumerate(pairs, start=1):
                    writer.writerow([i, t, f"{tf:.6f}", f"{kb:.6f}", f"{c:.6f}"])
            else:
                writer.writerow(["rank", "term", "score"])
                for i, (t, s) in enumerate(pairs, start=1):
                    writer.writerow([i, t, f"{s:.6f}"])
    elif fmt == "txt":
        terms_only = [t for t, *_ in pairs]
        lines = [
            f"# Generated by openalex extract-keywords from {source}",
            f"# params: {params}",
            *terms_only,
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise click.ClickException(f"Unknown --format '{fmt}'.")


@click.command("extract-keywords")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--title-col", default="Article Title", show_default=True)
@click.option("--abstract-col", default="Abstract", show_default=True)
@click.option("--top-n", type=int, default=50, show_default=True)
@click.option("--ngram-min", type=int, default=2, show_default=True)
@click.option("--ngram-max", type=int, default=3, show_default=True)
@click.option("--min-df", type=int, default=2, show_default=True)
@click.option("--max-df", type=float, default=0.85, show_default=True)
@click.option(
    "--use-keybert", is_flag=True,
    help="Re-score TF-IDF candidates with KeyBERT semantic similarity.",
)
@click.option(
    "--keybert-model", default="all-MiniLM-L6-v2", show_default=True,
    help="Sentence-transformers model name for KeyBERT.",
)
@click.option(
    "--candidate-pool", type=int, default=200, show_default=True,
    help="How many top TF-IDF terms to pass to KeyBERT as candidates (bigger = slower).",
)
@click.option(
    "--alpha", type=float, default=0.5, show_default=True,
    help="Blend weight: 1.0 = pure TF-IDF, 0.0 = pure KeyBERT.",
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv", "txt"], case_sensitive=False),
    default="json", show_default=True,
)
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--no-write", is_flag=True)
@click.option("--force", is_flag=True)
def extract_keywords_command(
    input_file: Path,
    title_col: str,
    abstract_col: str,
    top_n: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    use_keybert: bool,
    keybert_model: str,
    candidate_pool: int,
    alpha: float,
    fmt: str,
    output_path: Path | None,
    no_write: bool,
    force: bool,
) -> None:
    """Run TF-IDF (optionally + KeyBERT) over title+abstract in INPUT_FILE."""
    if ngram_min < 1 or ngram_max < ngram_min:
        raise click.ClickException("Require 1 <= --ngram-min <= --ngram-max.")
    if top_n < 1:
        raise click.ClickException("--top-n must be >= 1.")
    if use_keybert and not (0.0 <= alpha <= 1.0):
        raise click.ClickException("--alpha must be between 0.0 and 1.0.")

    fmt = fmt.lower()
    if output_path is None:
        output_path = Path(f"config/tfidf_keywords.{fmt}")

    df = _load_dataframe(input_file)
    docs = _build_documents(df, title_col, abstract_col)
    if not docs:
        console.print("[bold red]X No usable documents after filtering.[/bold red]")
        raise SystemExit(1)
    console.print(f"[dim]Loaded {len(docs)} documents from {input_file}[/dim]")

    pairs = _score_terms(docs, ngram_min, ngram_max, min_df, max_df)
    if not pairs:
        console.print("[bold red]X TF-IDF produced no terms (relax --min-df / --max-df).[/bold red]")
        raise SystemExit(1)

    params = {
        "top_n": top_n, "ngram_min": ngram_min, "ngram_max": ngram_max,
        "min_df": min_df, "max_df": max_df,
    }

    if use_keybert:
        # Candidate pool: take more than top_n from TF-IDF so KeyBERT has
        # room to re-rank (a term ranked #150 by TF-IDF might be #3 semantically).
        pool = pairs[:candidate_pool]
        candidate_terms = [t for t, _ in pool]
        kb_scores = _keybert_score(docs, candidate_terms, keybert_model)
        blended = _blend_scores(pool, kb_scores, alpha)
        top = blended[:top_n]
        _print_blended_table(top)
        params.update({
            "use_keybert": True, "keybert_model": keybert_model,
            "candidate_pool": candidate_pool, "alpha": alpha,
        })
    else:
        top = pairs[:top_n]
        _print_table(top)

    if no_write:
        return

    if not _confirm_overwrite(output_path, force):
        console.print("[yellow]Skipped writing output file.[/yellow]")
        return

    _write_output(top, output_path, fmt, input_file, params, blended=use_keybert)
    console.print(f"[green]OK Wrote {len(top)} keywords to [cyan]{output_path}[/cyan][/green]")