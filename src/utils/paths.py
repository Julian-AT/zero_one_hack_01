"""Shared filesystem paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = REPO_ROOT / "tracks" / "industrial-infineon"
RAW_DATA_DIR = TRACK_DIR / "training_data"

EXTRAS_DIR = REPO_ROOT / "extras"
CHECKPOINTS_DIR = EXTRAS_DIR / "checkpoints"
LOGS_DIR = EXTRAS_DIR / "logs"
RESULTS_DIR = EXTRAS_DIR / "results"
EDA_DIR = EXTRAS_DIR / "eda"

CONFIGS_DIR = REPO_ROOT / "configs"

FAMILY_FILES = {
    "mosfet": RAW_DATA_DIR / "MOSFET_variants.csv",
    "igbt": RAW_DATA_DIR / "IGBT_variants.csv",
    "ic": RAW_DATA_DIR / "IC_variants.csv",
}
FAMILY_PARAM_FILES = {
    "mosfet": RAW_DATA_DIR / "MOSFET_longdescription_parameters.csv",
    "igbt": RAW_DATA_DIR / "IGBT_longdescription_parameters.csv",
    "ic": RAW_DATA_DIR / "IC_longdescription_parameters.csv",
}
