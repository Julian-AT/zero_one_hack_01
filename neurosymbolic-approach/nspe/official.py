"""Locate and import the organizers' ground-truth modules by absolute path.

We never modify the organizers' code. We import:
  - generate_sequences.py  -> validator (`validate_sequence`) + grammar generator
  - generate_ood_families.py -> simulated unseen-family generators (for OOD tests)
  - eval_metrics.py is invoked as a subprocess (see nspe.eval), not imported here.

This mirrors how tracks/.../scripts/generate_ood_families.py loads the validator.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACK = REPO / "tracks" / "industrial-infineon"
GEN_PATH = TRACK / "training_data" / "generate_sequences.py"
OOD_PATH = TRACK / "scripts" / "generate_ood_families.py"
EVAL_PATH = TRACK / "scripts" / "eval_metrics.py"
DATA_DIR = TRACK / "training_data"
EVAL_DIR = TRACK / "scripts"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gs = _load("generate_sequences", GEN_PATH)        # official validator + grammar
ood = _load("generate_ood_families", OOD_PATH)    # unseen-family generators

validate_sequence = gs.validate_sequence
generate_sequence = gs.generate_sequence
read_csv_sequences = gs.read_csv_sequences
Violation = gs.Violation

FAMILIES = ("mosfet", "igbt", "ic")
FAMILY_FILES = {f: DATA_DIR / f"{f.upper()}_variants.csv" for f in FAMILIES}

# Sample eval inputs distributed with the track (used to dry-run the pipeline).
EVAL_INPUT_VALID = EVAL_DIR / "eval_input_valid.csv"
EVAL_INPUT_ANOMALY = EVAL_DIR / "eval_input_anomaly.csv"

__all__ = [
    "gs", "ood", "validate_sequence", "generate_sequence", "read_csv_sequences",
    "Violation", "REPO", "TRACK", "GEN_PATH", "OOD_PATH", "EVAL_PATH",
    "DATA_DIR", "EVAL_DIR", "FAMILIES", "FAMILY_FILES",
    "EVAL_INPUT_VALID", "EVAL_INPUT_ANOMALY",
]
