"""Planar pushing testbed modules: arm (iiwa), object, table, controller."""

from pathlib import Path

# Where generated SDFs land: written per-run, parsed right back.
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
