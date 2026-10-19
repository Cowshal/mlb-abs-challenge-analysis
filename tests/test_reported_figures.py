"""Keep the drift gate sensitive to changed statistical conclusions."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import reported_figures as rf


@pytest.mark.parametrize("key,written,current", [
    ("cin_bonferroni_p", 0.04, 0.06),
    ("player_split_half_ci_lo_min8", 0.02, -0.02),
])
def test_small_change_across_a_conclusion_threshold_fails(key, written, current):
    figure = next(f for f in rf.FIGURES if f.key == key)
    assert abs(written - current) < figure.tol
    problem = rf._check_one(figure, current, f"value={written}",
                            rf.Occ("README.md", r"value=([\d.\-]+)"))
    assert problem is not None
    assert "threshold" in problem.detail


def test_small_change_on_the_same_side_of_threshold_still_passes():
    figure = next(f for f in rf.FIGURES if f.key == "cin_bonferroni_p")
    assert rf._check_one(figure, 0.07414, "p=0.07",
                         rf.Occ("README.md", r"p=([\d.]+)")) is None


def test_gate_still_reports_a_missing_anchor():
    figure = next(f for f in rf.FIGURES if f.key == "cin_bonferroni_p")
    problem = rf._check_one(figure, 0.07414, "rewritten prose", figure.occ[0])
    assert problem is not None
    assert problem.line is None
