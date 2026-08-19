"""openalex impute-country — country imputation pipeline (Pass 0-C).

Standalone command. Unlike the (deprecated) `impute` subgroup, this is the
only imputation path currently maintained — kept as a top-level command
rather than nested under a group with sources that no longer apply.
"""

import click

from openalex.country_imputation_workflow import run_country_imputation


@click.command("impute-country")
@click.option(
    "--db-path",
    prompt="Path to database",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the .duckdb file to run country imputation against.",
)
@click.option(
    "--auto-write",
    is_flag=True,
    default=False,
    help=(
        "Commit Pass B (country-name match) and region-pass matches "
        "automatically. Without this flag, those passes only print "
        "candidate matches for review — nothing is written for them. "
        "Pass 0 (ROR API repair) and Pass A (institution join) always "
        "write, since they carry no ambiguity."
    ),
)
@click.option(
    "--ror-test-limit",
    type=int,
    default=None,
    help=(
        "Cap Pass 0 to the first N institutions needing a ROR API lookup. "
        "Use this for a quick validation run on a new dataset before "
        "committing to the full (potentially long) live-API batch."
    ),
)
def impute_country_command(db_path: str, auto_write: bool, ror_test_limit: int) -> None:
    """Recover missing country_code values in `contributions`.

    \b
    Runs four passes, most trustworthy first:
      0. ROR API institution repair — fixes institutions.country_code via
         the live ROR API (exact ID match, always writes).
      A. Join-fill from a matched institution_id (always writes).
      B. Country-name keyword match via pycountry (review-first by default;
         pass --auto-write to commit after reviewing the printed sample).
      C. Sub-national region matching — US states, Indian states (same
         review-first behavior as Pass B).

    \b
    Example:
      openalex impute-country --db-path data/db/mydata.duckdb
      openalex impute-country --db-path data/db/mydata.duckdb --auto-write
      openalex impute-country --db-path data/db/mydata.duckdb --ror-test-limit 10
    """
    summary = run_country_imputation(
        db_path,
        auto_write=auto_write,
        ror_test_limit=ror_test_limit,
    )

    click.echo("\n--- Summary ---")
    click.echo(f"Start nulls:  {summary['start_nulls']:,}")
    click.echo(f"End nulls:    {summary['end_nulls']:,}")
    click.echo(f"Recovered:    {summary['total_recovered']:,}")
    if not auto_write:
        click.echo(
            "\nPass B / region-pass matches were found but not written "
            "(review-only mode). Re-run with --auto-write to commit them."
        )