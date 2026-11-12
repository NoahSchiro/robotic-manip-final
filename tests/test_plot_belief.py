"""Belief plotting on a synthetic record: schema round trip and a rendered PNG."""

from pathlib import Path

import numpy as np
import pytest

from scripts.plot_belief import BeliefRecord, load_record, main, save_record
from src.object import Object, Rod
from src.params import Eta, GridSpec, make_grid

TRUE_ETA = Eta(mu=0.42, bx=0.012, by=-0.018)


def synthetic_record(pushes: int) -> BeliefRecord:
    """A prior and `pushes` posteriors that sharpen around TRUE_ETA, with
    contacts taken from the block's own outline sampling."""
    obj = Object(np.random.default_rng(0), rod=Rod(0.0, 0.0, 0.2))
    grid = make_grid(GridSpec(), obj.parts)
    hypotheses = np.array([eta.as_array() for eta in grid])
    true = TRUE_ETA.as_array()
    log_weights = np.zeros((pushes + 1, len(grid)))
    widths = [(0.15, 0.05), (0.08, 0.02), (0.04, 0.01), (0.03, 0.006)]
    for k in range(pushes):
        sigma_mu, sigma_xy = widths[k]
        log_weights[k + 1] = log_weights[k] - 0.5 * (
            ((hypotheses[:, 0] - true[0]) / sigma_mu) ** 2
            + (
                np.hypot(hypotheses[:, 1] - true[1], hypotheses[:, 2] - true[2])
                / sigma_xy
            )
            ** 2
        )
    probs = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    safe = np.where(probs > 0.0, probs, 1.0)
    entropy = -(probs * np.log(safe)).sum(axis=1)

    sites = obj.surface_normals(200.0)
    chosen = [sites[(i * len(sites)) // (pushes + 1)] for i in range(1, pushes + 1)]
    poses = np.zeros((pushes + 1, 3))
    poses[:, 0] = 0.55
    return BeliefRecord(
        grid=hypotheses,
        probs=probs,
        eta_true=true,
        parts=np.array(obj.parts, dtype=float),
        poses=poses,
        entropy=entropy,
        contact_position=np.array([s.position for s in chosen]).reshape(-1, 3),
        contact_normal=np.array([s.normal for s in chosen]).reshape(-1, 3),
        speed=np.full(pushes, 0.05),
        duration=np.full(pushes, 0.5),
        metadata={"seed": 123, "synthetic": True},
    )


def test_record_round_trips_through_npz(tmp_path: Path) -> None:
    record = synthetic_record(2)
    path = tmp_path / "belief.npz"
    save_record(path, record)
    loaded = load_record(path)
    np.testing.assert_array_equal(loaded.grid, record.grid)
    np.testing.assert_array_equal(loaded.probs, record.probs)
    np.testing.assert_array_equal(loaded.contact_normal, record.contact_normal)
    assert loaded.metadata["seed"] == 123
    assert loaded.metadata["schema_version"] == 1


@pytest.mark.parametrize("pushes", [0, 2])
def test_plot_writes_png(tmp_path: Path, pushes: int) -> None:
    record_path = tmp_path / "belief.npz"
    save_record(record_path, synthetic_record(pushes))
    out = tmp_path / "belief.png"
    assert main(["plot", str(record_path), "--out", str(out)]) == 0
    png = out.read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 10_000
