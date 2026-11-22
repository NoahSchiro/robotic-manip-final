"""Package E's condition names (no physics)."""

import pytest

from src.placement_eval import Condition, parse_condition


def test_conditions_parse() -> None:
    assert parse_condition("true_eta") == Condition("true_eta", None, 0, True)
    assert parse_condition("none") == Condition("none", None, 0, False)
    assert parse_condition("learned:3") == Condition("learned:3", "learned", 3, False)
    assert parse_condition("long_fast:5").prober == "long_fast"


@pytest.mark.parametrize("name", ["random", "random:0", "greedy:2", "learned:x", ""])
def test_bad_conditions_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        parse_condition(name)
