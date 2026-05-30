"""Shared CSV / sequence-string helpers for the eval and reranking scripts.

Stdlib-only on purpose: importing this module must never pull in torch, so the
prediction post-processing scripts (`competition/participant-files/*`) and baselines can use
it without paying for a heavy import. Previously these helpers were copy-pasted
across five scripts with subtly different behavior; this is the single home.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path


def norm(x: str) -> str:
    """Uppercase + strip a step/family token, tolerating None."""
    return str(x or "").strip().upper()


def split_steps(s: str, normalize: bool = True) -> list[str]:
    """Split a pipe-delimited sequence string into steps.

    Accepts both the `|||` and `|` delimiters (organizer eval files use `|`;
    some generated long-format exports use `|||`). With `normalize=True` each
    step is upper-cased; pass `normalize=False` to preserve original casing.
    """
    s = str(s or "").strip()
    if not s:
        return []
    sep = "|||" if "|||" in s else "|"
    parts = (x for x in s.split(sep) if x.strip())
    return [norm(x) if normalize else x.strip() for x in parts]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts (BOM-tolerant)."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalized_row(row: dict[str, str]) -> dict[str, str]:
    """Strip BOM/whitespace/quote noise from a raw CSV row's keys and values."""
    return {
        str(k).strip().lstrip("\ufeff").strip('"'): (v or "").strip().strip('"')
        for k, v in row.items()
    }


def iter_grouped_sequences(
    path: Path, warn_missing: bool = False
) -> Iterator[tuple[str, str, list[str], dict[str, str] | None]]:
    """Stream a long-format sequence CSV grouped by SEQUENCE_ID.

    Expects columns SEQUENCE_ID, FAMILY, STEP (extra columns preserved on the
    first row of each group). Yields one tuple per sequence:

        (sequence_id, family, steps, first_row)

    `first_row` is the normalized first row of the group, exposing per-sequence
    metadata such as MUTATION_INDEX. Callers that only need the steps can ignore
    it. Rows missing a SEQUENCE_ID or STEP are skipped.
    """
    if not path.exists():
        if warn_missing:
            print(f"[WARN] missing sequence source: {path}")
        return

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        current_id: str | None = None
        current_family: str | None = None
        current_steps: list[str] = []
        first_row: dict[str, str] | None = None

        for raw in reader:
            row = normalized_row(raw)
            sid = row.get("SEQUENCE_ID", "")
            fam = norm(row.get("FAMILY", "UNKNOWN"))
            step = norm(row.get("STEP", ""))

            if not sid or not step:
                continue

            if current_id is None:
                current_id, current_family, current_steps, first_row = sid, fam, [], row
            elif sid != current_id:
                yield current_id, current_family, current_steps, first_row
                current_id, current_family, current_steps, first_row = sid, fam, [], row

            current_steps.append(step)

        if current_id is not None:
            yield current_id, current_family, current_steps, first_row
