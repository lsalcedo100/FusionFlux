"""The summary figure must stay consistent with the artifacts it summarises.

``analysis_summary_figure.py`` is the one analysis script that reads only
generated artifacts and never the raw database, which makes it the one that can
go stale without failing: it will happily draw last week's numbers if the
artifacts move and nobody reruns it. These tests bind what it draws to what the
artifacts say, and check that the two panels are reading the fields they claim
to be reading.

The figure itself is not compared byte-for-byte. Matplotlib output is not
reproducible across versions, and the ``reproduce`` workflow excludes figures
from its comparison for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import analysis_summary_figure as summary_figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def artifacts() -> tuple[pd.DataFrame, dict]:
    return summary_figure._read()


def test_the_figure_is_committed() -> None:
    """It is referenced from the README, so a missing file is a broken image."""
    assert (RESULTS / "summary.png").exists(), "run `python3 analysis_summary_figure.py`"


def test_every_model_it_draws_exists_in_both_artifacts(
    artifacts: tuple[pd.DataFrame, dict],
) -> None:
    """A typo in STYLE drops a model silently rather than raising."""
    scores, forecast = artifacts
    forecast_models = {f["model_name"] for f in forecast["forecasts"]}
    drawn_left = [m for m in summary_figure.STYLE if m != "powerlaw_collisionless"]
    assert set(drawn_left) <= set(scores.index)
    assert set(summary_figure.STYLE) <= forecast_models


def test_the_left_panel_actually_shows_a_reversal(
    artifacts: tuple[pd.DataFrame, dict],
) -> None:
    """The title claims the best model under one split is worst under the other.

    If a rerun ever made that false, the figure would keep asserting it in
    12.5pt type. This is the assertion behind the title.
    """
    scores, _ = artifacts
    # Restricted to the models the panel draws. ``extrapolation_summary.csv``
    # also carries ``mean_baseline``, the constant predictor, which is worst
    # under every split by construction and would make this pass for the wrong
    # reason; and IPB98(y,2) is excluded because it is not blind.
    drawn = [
        m for m in summary_figure.STYLE
        if m in scores.index and m != "ipb98y2_analytic"
    ]
    blind = scores.loc[drawn]
    best_cv = blind["cv_rmsle"].idxmin()
    worst_lomo = blind["lomo_mean_rmsle"].idxmax()
    assert best_cv == worst_lomo, (
        f"the figure's title says the best CV model is the worst on a new machine, "
        f"but they are now {best_cv!r} and {worst_lomo!r}"
    )


def test_the_right_panel_title_matches_the_forecast(
    artifacts: tuple[pd.DataFrame, dict],
) -> None:
    """The spread in the title is computed, and must match the prose's 8.3."""
    _, forecast = artifacts
    taus = [f["tau_predicted_s"] for f in forecast["forecasts"] if f["device"] == "ITER"]
    assert f"{max(taus) / min(taus):.1f}" == "8.3"


def test_the_bounded_models_are_the_ones_under_the_ceiling(
    artifacts: tuple[pd.DataFrame, dict],
) -> None:
    """The 'cannot exceed the ceiling' caption is drawn off a flag, not a guess.

    So the flag has to agree with the arithmetic it stands for: a model marked
    bounded must predict at or below the largest training target, and one not
    marked must not be.
    """
    _, forecast = artifacts
    ceiling = float(forecast["train_tau_max_s"])
    for row in forecast["forecasts"]:
        if row["device"] != "ITER":
            continue
        if row["bounded_by_training_range"]:
            assert row["tau_predicted_s"] <= ceiling
        else:
            # The unbounded models are the ones that clear the ceiling at ITER;
            # that contrast is the whole content of the right-hand panel.
            assert row["tau_predicted_s"] > ceiling


def test_it_runs_and_writes_the_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end, into a temporary directory rather than over the committed file."""
    pytest.importorskip("matplotlib")
    staging = tmp_path / "results"
    staging.mkdir()
    for name in ("extrapolation_summary.csv", "forecast.json"):
        (staging / name).write_bytes((RESULTS / name).read_bytes())
    monkeypatch.setattr(summary_figure, "RESULTS_DIR", staging)

    path = summary_figure.plot_summary()
    assert path is not None
    assert path == staging / "summary.png"
    assert path.stat().st_size > 10_000
