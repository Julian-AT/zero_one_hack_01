"""Tests for the shared CSV / sequence-string helpers."""

from __future__ import annotations

from pathlib import Path

from src.data.sequence_io import (
    iter_grouped_sequences,
    norm,
    normalized_row,
    read_csv,
    split_steps,
)


def test_norm_uppercases_and_strips():
    assert norm("  deposit oxide  ") == "DEPOSIT OXIDE"
    assert norm("") == ""
    assert norm(None) == ""


def test_split_steps_pipe_delimited_normalized():
    assert split_steps("a|b|c") == ["A", "B", "C"]


def test_split_steps_triple_pipe_takes_precedence():
    # When both delimiters could match, the triple-pipe wins.
    assert split_steps("a|b|||c|d") == ["A|B", "C|D"]


def test_split_steps_preserves_case_when_not_normalized():
    assert split_steps("Deposit|Etch", normalize=False) == ["Deposit", "Etch"]


def test_split_steps_empty_and_whitespace():
    assert split_steps("") == []
    assert split_steps("   ") == []
    assert split_steps(None) == []
    # Empty segments are dropped.
    assert split_steps("a||b") == ["A", "B"]


def test_read_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "rows.csv"
    p.write_text("EXAMPLE_ID,FAMILY\n1,MOSFET\n2,IGBT\n", encoding="utf-8")
    rows = read_csv(p)
    assert rows == [
        {"EXAMPLE_ID": "1", "FAMILY": "MOSFET"},
        {"EXAMPLE_ID": "2", "FAMILY": "IGBT"},
    ]


def test_read_csv_strips_bom(tmp_path: Path):
    p = tmp_path / "bom.csv"
    p.write_bytes("\ufeffA,B\n1,2\n".encode())
    rows = read_csv(p)
    assert list(rows[0].keys()) == ["A", "B"]


def test_read_csv_missing_raises(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    try:
        read_csv(missing)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_normalized_row_strips_bom_and_quotes():
    raw = {"\ufeffSEQUENCE_ID": '"42"', "STEP": "  etch  "}
    cleaned = normalized_row(raw)
    assert cleaned == {"SEQUENCE_ID": "42", "STEP": "etch"}


def _write_long_csv(path: Path) -> None:
    path.write_text(
        "SEQUENCE_ID,FAMILY,STEP,MUTATION_INDEX\n"
        "s1,MOSFET,DEPOSIT OXIDE,2\n"
        "s1,MOSFET,ETCH OXIDE,2\n"
        "s2,IGBT,CLEAN WAFER,\n"
        "s2,IGBT,DEPOSIT METAL,\n"
        "s2,IGBT,CMP METAL,\n",
        encoding="utf-8",
    )


def test_iter_grouped_sequences_groups_by_id(tmp_path: Path):
    p = tmp_path / "long.csv"
    _write_long_csv(p)
    groups = list(iter_grouped_sequences(p))
    assert [g[0] for g in groups] == ["s1", "s2"]
    assert [g[1] for g in groups] == ["MOSFET", "IGBT"]
    assert groups[0][2] == ["DEPOSIT OXIDE", "ETCH OXIDE"]
    assert groups[1][2] == ["CLEAN WAFER", "DEPOSIT METAL", "CMP METAL"]
    # first_row exposes per-sequence metadata.
    assert groups[0][3]["MUTATION_INDEX"] == "2"


def test_iter_grouped_sequences_skips_rows_without_id_or_step(tmp_path: Path):
    p = tmp_path / "gappy.csv"
    p.write_text(
        "SEQUENCE_ID,FAMILY,STEP\n"
        "s1,MOSFET,DEPOSIT OXIDE\n"
        ",MOSFET,ORPHAN STEP\n"
        "s1,MOSFET,\n"
        "s1,MOSFET,ETCH OXIDE\n",
        encoding="utf-8",
    )
    groups = list(iter_grouped_sequences(p))
    assert len(groups) == 1
    assert groups[0][2] == ["DEPOSIT OXIDE", "ETCH OXIDE"]


def test_iter_grouped_sequences_missing_file_yields_nothing(tmp_path: Path):
    assert list(iter_grouped_sequences(tmp_path / "nope.csv")) == []
