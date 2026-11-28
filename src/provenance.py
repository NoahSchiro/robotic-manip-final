"""The code version an output was produced with, for result records."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state() -> dict[str, object]:
    """The checkout's commit and whether it has uncommitted changes; commit
    "unknown" and dirty None outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except OSError, subprocess.CalledProcessError:
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}
