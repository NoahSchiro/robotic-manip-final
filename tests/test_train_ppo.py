"""PPO run bookkeeping: resuming and the learning-curve log (no training)."""

import json
from pathlib import Path

from src.train_ppo import CURVE_FILE, EvalPoint, latest_checkpoint, load_curve


def test_latest_checkpoint_is_the_one_with_the_most_steps(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    for name in ("ppo_100000.zip", "ppo_900000.zip", "ppo_final.zip", "ppo_20000.zip"):
        (tmp_path / name).write_text("")
    assert latest_checkpoint(tmp_path) == tmp_path / "ppo_900000.zip"


def test_learning_curve_reloads_logged_points(tmp_path: Path) -> None:
    assert load_curve(tmp_path) == []
    point = EvalPoint(100000, -0.5, -0.7, 0.1, 0.2, 4.0)
    (tmp_path / CURVE_FILE).write_text(json.dumps(point.__dict__) + "\n")
    assert load_curve(tmp_path) == [point]
