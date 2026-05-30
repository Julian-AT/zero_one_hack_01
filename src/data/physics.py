"""Parse physics parameters from `*_longdescription_parameters.csv` into a
step_string → 10-dimensional feature vector lookup.

Features (a fixed schema; missing values get a sentinel `nan`/`-1`):
    0  temp_C                 (°C)                — float / NaN
    1  log_time_s             log10(seconds)      — float / NaN
    2  log_thickness_nm       log10(nm)           — float / NaN
    3  log_pressure_torr      log10(Torr)         — float / NaN
    4  energy_keV                                  — float / NaN
    5  log_dose_per_cm2       log10(dose)         — float / NaN
    6  tool_category          int 0..9            — categorical
    7  is_wet                 bool                — 0/1
    8  is_anneal              bool                — 0/1
    9  is_implant_or_dope     bool                — 0/1

The numeric features are roughly standardized inside reasonable ranges so a
simple `Linear(10, d_model)` projection added to the token embedding behaves
well.

Why this exists: cell 4 OOD generalization on the hidden family. A step
string we have never seen can still be embedded via its parameters (e.g.
"LPCVD; 620 °C; thickness 400 nm" places it near DEPOSIT POLYSILICON).

CLI:
    python -m src.data.physics              # build lookup, save to data/processed/
    python -m src.data.physics --inspect    # print a few examples
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from src.utils.paths import FAMILY_PARAM_FILES, RAW_DATA_DIR

# Ensure we can read the organizers' CSV.
if str(RAW_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(RAW_DATA_DIR))

# We do NOT use read_csv_sequences (which expects SEQUENCE_ID/STEP columns).
# The longdescription_parameters files have STEP, DESCRIPTION, REALISTIC FAB‑LEVEL PARAMETERS.
import csv  # noqa: E402

NAN = float("nan")

# Translate unicode superscripts/subscripts to ASCII for cleaner regex.
_SUPERSCRIPTS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)

TOOL_TAXONOMY = [
    "LPCVD",
    "PECVD",
    "RPCVD",
    "PVD",
    "ICP",
    "RIE",
    "CMP",
    "WET",  # wet etch / wet clean
    "FURNACE",  # furnace anneal / oxidation
    "OTHER",
]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOL_TAXONOMY)}


_TEMP_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*°\s*C")
_ENERGY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*keV")
_PRESSURE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mTorr|Torr)", re.IGNORECASE)
_THICK_NM_RE = re.compile(r"(?:thickness\s*)?(\d+(?:[.,]\d+)?)(?:[–\-](\d+(?:[.,]\d+)?))?\s*nm")
_THICK_UM_RE = re.compile(r"(?:thickness\s*)?(\d+(?:[.,]\d+)?)(?:[–\-](\d+(?:[.,]\d+)?))?\s*µm")
_DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[x×*]\s*10\s*\^?\s*(-?\d+)\s*cm\s*-?\s*\d?", re.UNICODE)
# Time units
_TIME_S_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*s\b")
_TIME_MIN_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*min\b")
_TIME_HR_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*hr\b")


def _f(s: str) -> float:
    """Parse a possibly-comma decimal."""
    return float(s.replace(",", "."))


def _midpoint(m: re.Match) -> float:
    """For a range like "60–120 nm" return the midpoint; else the single value."""
    lo = _f(m.group(1))
    if m.lastindex and m.lastindex >= 2 and m.group(2):
        hi = _f(m.group(2))
        return (lo + hi) / 2
    return lo


def parse_one(params: str, description: str = "") -> list[float]:
    """Parse one parameter string into a 10-d feature vector.

    Returns NaN / -1 for fields not found.
    """
    p = (params or "").translate(_SUPERSCRIPTS)
    description = (description or "").translate(_SUPERSCRIPTS)

    m = _TEMP_RE.search(p)
    temp_c = _f(m.group(1)) if m else NAN

    secs: float | None = None
    m = _TIME_HR_RE.search(p)
    if m:
        secs = _f(m.group(1)) * 3600
    if secs is None:
        m = _TIME_MIN_RE.search(p)
        if m:
            secs = _f(m.group(1)) * 60
    if secs is None:
        m = _TIME_S_RE.search(p)
        if m:
            secs = _f(m.group(1))
    log_time_s = math.log10(secs) if (secs is not None and secs > 0) else NAN

    thick_nm: float | None = None
    m = _THICK_NM_RE.search(p)
    if m:
        thick_nm = _midpoint(m)
    if thick_nm is None:
        m = _THICK_UM_RE.search(p)
        if m:
            thick_nm = _midpoint(m) * 1000  # µm → nm
    log_thickness_nm = math.log10(thick_nm) if (thick_nm and thick_nm > 0) else NAN

    log_pressure = NAN
    m = _PRESSURE_RE.search(p)
    if m:
        v = _f(m.group(1))
        unit = m.group(2).lower()
        if unit == "mtorr":
            v = v / 1000.0
        if v > 0:
            log_pressure = math.log10(v)

    m = _ENERGY_RE.search(p)
    energy_keV = _f(m.group(1)) if m else NAN

    log_dose = NAN
    m = _DOSE_RE.search(p)
    if m:
        coef = _f(m.group(1))
        expo = int(m.group(2))
        if coef > 0:
            log_dose = math.log10(coef) + expo

    up = p.upper() + " " + (description or "").upper()
    tool_idx = TOOL_TO_IDX["OTHER"]
    for t in ["LPCVD", "PECVD", "RPCVD", "PVD", "ICP", "RIE", "CMP"]:
        if t in up:
            tool_idx = TOOL_TO_IDX[t]
            break
    else:
        if any(
            k in up
            for k in ["WET CLEAN", "RCA", "HF", "WET ETCH", "PIRANHA", "SPM", "DI ", "MEGASONIC"]
        ):
            tool_idx = TOOL_TO_IDX["WET"]
        elif any(k in up for k in ["FURNACE", "ANNEAL", "RTA", "RAPID THERMAL", "OXIDATION"]):
            tool_idx = TOOL_TO_IDX["FURNACE"]

    is_wet = 1.0 if tool_idx == TOOL_TO_IDX["WET"] else 0.0

    is_anneal = 1.0 if any(k in up for k in ["ANNEAL", "RTA", "RAPID THERMAL"]) else 0.0

    is_dope = 1.0 if any(k in up for k in ["IMPLANT", "DOPE", "DOSE", " KEV"]) else 0.0

    return [
        temp_c,
        log_time_s,
        log_thickness_nm,
        log_pressure,
        energy_keV,
        log_dose,
        float(tool_idx),
        is_wet,
        is_anneal,
        is_dope,
    ]


def _read_param_csv(path: Path) -> list[tuple[str, str, str]]:
    """Read STEP, DESCRIPTION, REALISTIC FAB-LEVEL PARAMETERS columns."""
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # column names sometimes have ‑ vs - vs spacing differences
            step = (row.get("STEP") or row.get("Step") or "").strip().strip('"')
            desc = (row.get("DESCRIPTION") or row.get("Description") or "").strip().strip('"')
            params_key = next((k for k in row if "PARAM" in k.upper()), None)
            params = (row[params_key] if params_key else "").strip().strip('"')
            if step:
                rows.append((step, desc, params))
    return rows


def build_lookup() -> dict[str, list[float]]:
    """Build step_string → 10-d feature vector by merging info from all 3
    family parameter CSVs. When a step appears in multiple families, we use
    the first non-NaN value per feature (good enough for our use)."""
    merged: dict[str, list[float]] = {}
    sources: dict[str, list[str]] = {}
    for fam, path in FAMILY_PARAM_FILES.items():
        for step, desc, params in _read_param_csv(Path(path)):
            v = parse_one(params, desc)
            if step not in merged:
                merged[step] = v[:]
                sources[step] = [fam]
            else:
                # Merge: prefer non-NaN values from later families.
                for i in range(len(v)):
                    if math.isnan(merged[step][i]) and not math.isnan(v[i]):
                        merged[step][i] = v[i]
                sources[step].append(fam)
    return merged


def features_stats(lookup: dict[str, list[float]]) -> dict:
    """Sanity stats on the lookup."""
    feats = list(lookup.values())
    n = len(feats)
    names = [
        "temp_C",
        "log_time_s",
        "log_thickness_nm",
        "log_pressure_torr",
        "energy_keV",
        "log_dose",
        "tool_idx",
        "is_wet",
        "is_anneal",
        "is_implant",
    ]
    stats: dict = {"n_steps": n}
    for i, name in enumerate(names):
        vals = [v[i] for v in feats if not math.isnan(v[i])]
        if not vals:
            stats[name] = {"present": 0}
            continue
        stats[name] = {
            "present": len(vals),
            "present_pct": round(100 * len(vals) / n, 1),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "mean": round(sum(vals) / len(vals), 3),
        }
    return stats


if __name__ == "__main__":
    import argparse

    from src.utils.paths import REPO_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--out", default="data/processed/physics_features.json")
    args = parser.parse_args()

    lookup = build_lookup()
    out_path = Path(REPO_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the lookup as JSON for easy inspection / loading.
    with out_path.open("w") as f:
        json.dump(
            {
                "schema": [
                    "temp_C",
                    "log_time_s",
                    "log_thickness_nm",
                    "log_pressure_torr",
                    "energy_keV",
                    "log_dose",
                    "tool_idx",
                    "is_wet",
                    "is_anneal",
                    "is_implant_or_dope",
                ],
                "tool_taxonomy": TOOL_TAXONOMY,
                "lookup": lookup,
            },
            f,
            indent=2,
        )

    stats = features_stats(lookup)
    print(f"Built lookup for {stats['n_steps']} unique step strings.")
    for k, v in stats.items():
        if k == "n_steps":
            continue
        if "present_pct" in v:
            print(
                f"  {k:25s} present={v['present_pct']:5.1f}%  "
                f"min={v['min']:>9.3f}  mean={v['mean']:>9.3f}  max={v['max']:>9.3f}"
            )
        else:
            print(f"  {k:25s} present=0")

    if args.inspect:
        print("\nExamples:")
        for step in [
            "THERMAL OXIDATION",
            "DEPOSIT POLYSILICON",
            "IMPLANT P BODY",
            "STRIP PHOTORESIST",
            "WAFER SORT TEST",
            "RCA CLEAN 1",
        ]:
            v = lookup.get(step)
            print(f"  {step:30s} → {v}")

    print(f"\nSaved → {out_path}")
