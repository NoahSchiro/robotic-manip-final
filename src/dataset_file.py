"""The .npz dataset file format shared by the recorded-trial and surrogate
datasets: named arrays plus meta_-prefixed generation settings."""

from pathlib import Path

import numpy as np

META_PREFIX = "meta_"


def save_dataset(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, float | int | np.ndarray],
) -> None:
    """Write `arrays` with meta_-prefixed `metadata` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        allow_pickle=True,
        **arrays,
        **{f"{META_PREFIX}{key}": np.asarray(value) for key, value in metadata.items()},
    )
