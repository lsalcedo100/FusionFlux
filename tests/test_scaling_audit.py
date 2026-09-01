"""Tests for the reusable audit in ``scaling_audit.py``.

This module makes two kinds of claim and they need different evidence.

The first is that it is *correct*: the constrained solver has to actually
satisfy the constraint, and it has to agree with the from-scratch solver in
``scaling_law.py`` that the study's own results were computed with. A second
implementation that quietly disagrees with the first is worse than no second
implementation, so the equivalence is asserted rather than assumed.

The second is that it is *domain-agnostic*, which is a claim about what it does
not depend on. Asserting that is awkward, because the obvious test -- run it on
the tokamak data -- is exactly the test that cannot distinguish a general tool
from a specialised one. So the bulk of what follows runs on a synthetic
allometric problem with no plasma physics in it: groups are species clades,
the target is metabolic rate, and the generating law is a power law with a
clade-specific offset. If the audit recovers the same qualitative finding there
-- a flexible model that wins within groups and loses on an unseen one -- then
the finding is about the split rather than about fusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import scaling_audit as sa
import scaling_law as sl

# ---------------------------------------------------------------------------
# A synthetic problem from a different field entirely.
# ---------------------------------------------------------------------------
CLADES = ("shrews", "rodents", "carnivores", "ungulates", "cetaceans")


@pytest.fixture(scope="module")
def allometry() -> pd.DataFrame:
    """Metabolic rate against body mass, in logs, grouped by clade.

    Kleiber's law says metabolic rate scales as mass to the 3/4, which is a
    power law with a published exponent fitted across taxa: structurally the
    same object as a confinement scaling law, and chosen here for that reason.
    Each clade occupies its own stretch of the mass axis, and the stretches are
    ordered, so holding out ``cetaceans`` asks for masses above anything in
    training exactly as holding out JET asks for confinement times above
    anything in training.

    The per-clade offsets are small and deliberately *not* monotonic in the
    ordering. A clade effect that grew with mass would be a trend the features
    cannot see, and then the linear model's error would climb with distance too
    -- which is precisely the behaviour the real power law does not show
    (rho = -0.06). Making the offsets a nuisance rather than a trend is what
    lets this fixture reproduce the real contrast instead of muddying it.
    """
    rng = np.random.default_rng(20260901)
    offsets = (0.04, -0.03, 0.05, -0.02, 0.03)
    frames = []
    for index, clade in enumerate(CLADES):
        n = 120
        log_mass = rng.uniform(index * 1.6, index * 1.6 + 2.0, size=n)
        log_temperature = rng.normal(0.0, 0.3, size=n)
        offset = offsets[index]
        log_rate = 0.75 * log_mass + 0.1 * log_temperature + offset + rng.normal(0.0, 0.05, size=n)
        frames.append(
            pd.DataFrame(
                {
                    "clade": clade,
                    "log_mass": log_mass,
                    "log_temperature": log_temperature,
                    "log_rate": log_rate,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


FEATURES = ["log_mass", "log_temperature"]


# ---------------------------------------------------------------------------
# The constrained estimator.
# ---------------------------------------------------------------------------
def test_unconstrained_matches_ordinary_least_squares(allometry: pd.DataFrame) -> None:
    """With no constraint it must be plain OLS, or the control arm is not a control."""
    model = sa.ConstrainedLinearRegression().fit(allometry[FEATURES], allometry["log_rate"])
    reference = LinearRegression().fit(allometry[FEATURES], allometry["log_rate"])
    np.testing.assert_allclose(model.coef_, reference.coef_, rtol=1e-9, atol=1e-11)
    assert model.intercept_ == pytest.approx(float(reference.intercept_), rel=1e-9, abs=1e-11)


def test_the_constraint_is_actually_satisfied(allometry: pd.DataFrame) -> None:
    """A constraint that is silently not applied looks like one that did not help."""
    # Pin the mass exponent to Kleiber's 3/4. Columns are [intercept, mass, temp].
    constraint = np.array([[0.0, 1.0, 0.0]])
    model = sa.ConstrainedLinearRegression(constraint, np.array([0.75])).fit(
        allometry[FEATURES], allometry["log_rate"]
    )
    assert model.constraint_violation() < 1e-9
    assert model.coef_[0] == pytest.approx(0.75, abs=1e-9)


def test_constraint_costs_something_in_sample_and_nothing_is_free(
    allometry: pd.DataFrame,
) -> None:
    """The constrained fit cannot beat the unconstrained one on the training data.

    This is the sanity check that makes a later out-of-sample *gain* meaningful:
    it must be paid for somewhere, and in sample is where.
    """
    y = allometry["log_rate"].to_numpy()
    free = sa.ConstrainedLinearRegression().fit(allometry[FEATURES], y)
    pinned = sa.ConstrainedLinearRegression(
        np.array([[0.0, 1.0, 0.0]]), np.array([0.70])
    ).fit(allometry[FEATURES], y)
    free_rmse = float(np.sqrt(np.mean((y - free.predict(allometry[FEATURES])) ** 2)))
    pinned_rmse = float(np.sqrt(np.mean((y - pinned.predict(allometry[FEATURES])) ** 2)))
    assert pinned_rmse >= free_rmse - 1e-12


def test_agrees_with_the_from_scratch_solver(allometry: pd.DataFrame) -> None:
    """The duplicated KKT solve must match ``scaling_law``'s to numerical noise.

    ``scaling_audit`` carries its own copy so it can be lifted out of the
    repository standalone. This is what stops the copy drifting.
    """
    design = np.column_stack([np.ones(len(allometry)), allometry[FEATURES].to_numpy(dtype=float)])
    target = allometry["log_rate"].to_numpy(dtype=float)
    constraint = np.array([[0.0, 1.0, 0.0]])
    rhs = np.array([0.75])

    theirs = sl.solve_constrained_lstsq(design, target, constraint, rhs)
    ours = sa.ConstrainedLinearRegression(constraint, rhs).fit(
        allometry[FEATURES], target
    ).coefficients_
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-11)


def test_constraint_shape_is_checked_against_the_intercept_column(
    allometry: pd.DataFrame,
) -> None:
    """Forgetting the intercept column is the easy mistake; it must not be silent."""
    with pytest.raises(ValueError, match="columns but the design has"):
        sa.ConstrainedLinearRegression(np.array([[1.0, 0.0]]), np.array([0.75])).fit(
            allometry[FEATURES], allometry["log_rate"]
        )


def test_rhs_length_is_checked(allometry: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="rows but rhs has"):
        sa.ConstrainedLinearRegression(np.array([[0.0, 1.0, 0.0]]), np.array([0.1, 0.2])).fit(
            allometry[FEATURES], allometry["log_rate"]
        )


# ---------------------------------------------------------------------------
# The diagnostics.
# ---------------------------------------------------------------------------
def test_diagnostic_detects_targets_beyond_the_training_range(
    allometry: pd.DataFrame,
) -> None:
    """Holding out the largest clade must report target headroom, not zero."""
    labels = allometry["clade"].to_numpy()
    held = labels == "cetaceans"
    diagnostic = sa.group_diagnostic(
        allometry[FEATURES],
        allometry["log_rate"],
        np.flatnonzero(~held),
        np.flatnonzero(held),
        group="cetaceans",
    )
    assert diagnostic.fraction_above_train_max > 0.5
    assert diagnostic.log_target_headroom > 0.0
    assert diagnostic.n_held_out_rows == int(held.sum())


def test_diagnostic_reports_no_headroom_for_an_interior_group(
    allometry: pd.DataFrame,
) -> None:
    """The negative control: a middle clade is surrounded, so nothing is beyond."""
    labels = allometry["clade"].to_numpy()
    held = labels == "carnivores"
    diagnostic = sa.group_diagnostic(
        allometry[FEATURES],
        allometry["log_rate"],
        np.flatnonzero(~held),
        np.flatnonzero(held),
        group="carnivores",
    )
    assert diagnostic.fraction_above_train_max == 0.0
    assert diagnostic.log_target_headroom < 0.0


def test_diagnostic_survives_a_rank_deficient_feature_matrix(
    allometry: pd.DataFrame,
) -> None:
    """A derived feature makes the covariance singular; the pseudo-inverse must hold.

    This is not hypothetical: the study's own design matrix is rank deficient by
    two for exactly this reason.
    """
    frame = allometry.copy()
    frame["log_mass_copy"] = frame["log_mass"]
    columns = [*FEATURES, "log_mass_copy"]
    held = (frame["clade"] == "cetaceans").to_numpy()
    diagnostic = sa.group_diagnostic(
        frame[columns],
        frame["log_rate"],
        np.flatnonzero(~held),
        np.flatnonzero(held),
    )
    assert np.isfinite(diagnostic.mahalanobis)
    assert diagnostic.mahalanobis > 0.0


# ---------------------------------------------------------------------------
# The audit, on a problem with no physics in it.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report(allometry: pd.DataFrame) -> pd.DataFrame:
    return sa.audit_groups(
        allometry[FEATURES],
        allometry["log_rate"].to_numpy(),
        allometry["clade"],
        {
            "linear": sa.ConstrainedLinearRegression(),
            "forest": RandomForestRegressor(n_estimators=60, random_state=0),
        },
    )


def test_audit_returns_a_row_per_group_and_estimator(report: pd.DataFrame) -> None:
    assert set(report["group"]) == set(CLADES)
    assert set(report["estimator"]) == {"linear", "forest"}
    assert len(report) == len(CLADES) * 2


def test_the_reversal_reproduces_outside_fusion(report: pd.DataFrame) -> None:
    """The load-bearing claim of this module.

    A random forest is the more flexible model and fits within a clade better.
    Asked for a clade it has never seen, it must lose to the linear law, for the
    same structural reason it loses to a power law on a new tokamak. If this
    fails, the study's finding really was about the tokamak database.
    """
    worst = report.groupby("estimator")["score"].max()
    assert worst["forest"] > worst["linear"]

    per_group = report.pivot(index="group", columns="estimator", values="score")
    forest_loses = int((per_group["forest"] > per_group["linear"]).sum())
    assert forest_loses >= len(CLADES) - 1, per_group


def test_the_forest_is_bounded_and_the_linear_model_is_not(report: pd.DataFrame) -> None:
    """The hard bound, detected per fold rather than inferred from the model type."""
    extrapolating = report[report["fraction_above_train_max"] > 0.5]
    assert not extrapolating.empty, "no fold actually extrapolated; the fixture is wrong"
    forest = extrapolating[extrapolating["estimator"] == "forest"]
    linear = extrapolating[extrapolating["estimator"] == "linear"]
    assert forest["prediction_bounded_by_train_range"].all()
    assert not linear["prediction_bounded_by_train_range"].any()


def test_error_tracks_distance_for_the_flexible_model(report: pd.DataFrame) -> None:
    """The forest should fail as a function of how far out the group is."""
    correlation = sa.distance_score_correlation(report)
    assert correlation["forest"] > correlation["linear"]


def test_summarize_ranks_by_the_tail(report: pd.DataFrame) -> None:
    summary = sa.summarize(report)
    assert list(summary.index)[0] == "linear"
    assert summary.loc["forest", "worst"] >= summary.loc["forest", "median"]


def test_small_groups_are_skipped_rather_than_averaged_in(allometry: pd.DataFrame) -> None:
    frame = pd.concat(
        [allometry, allometry.head(2).assign(clade="fossil")], ignore_index=True
    )
    audited = sa.audit_groups(
        frame[FEATURES],
        frame["log_rate"].to_numpy(),
        frame["clade"],
        {"linear": sa.ConstrainedLinearRegression()},
        min_held_out_rows=10,
    )
    assert "fossil" not in set(audited["group"])


def test_mismatched_lengths_are_rejected(allometry: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must agree in length"):
        sa.audit_groups(
            allometry[FEATURES],
            allometry["log_rate"].to_numpy()[:-1],
            allometry["clade"],
            {"linear": sa.ConstrainedLinearRegression()},
        )


def test_an_empty_estimator_map_is_rejected(allometry: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one estimator"):
        sa.audit_groups(
            allometry[FEATURES], allometry["log_rate"].to_numpy(), allometry["clade"], {}
        )


def test_estimators_are_cloned_not_mutated(allometry: pd.DataFrame) -> None:
    """A caller's estimator must come back unfitted, or folds leak into each other."""
    estimator = sa.ConstrainedLinearRegression()
    sa.audit_groups(
        allometry[FEATURES],
        allometry["log_rate"].to_numpy(),
        allometry["clade"],
        {"linear": estimator},
    )
    assert not hasattr(estimator, "coefficients_")


# ---------------------------------------------------------------------------
# The ordered split.
# ---------------------------------------------------------------------------
def test_ordered_split_trains_on_the_small_end_and_predicts_the_large(
    allometry: pd.DataFrame,
) -> None:
    order = {clade: index for index, clade in enumerate(CLADES)}
    splitter = sa.OrderedGroupSplit(order, min_train_groups=2)
    labels = allometry["clade"].to_numpy()

    cuts = list(splitter.split(allometry[FEATURES], allometry["log_rate"], labels))
    assert len(cuts) == len(CLADES) - 2
    assert splitter.get_n_splits(groups=labels) == len(cuts)

    for train_index, test_index in cuts:
        train_groups = {order[g] for g in np.unique(labels[train_index])}
        test_groups = {order[g] for g in np.unique(labels[test_index])}
        # Every test group must rank above every training group; that is the
        # whole difference from leave-one-group-out.
        assert max(train_groups) < min(test_groups)
        assert not set(train_index) & set(test_index)


def test_ordered_split_cuts_grow_monotonically(allometry: pd.DataFrame) -> None:
    order = {clade: index for index, clade in enumerate(CLADES)}
    labels = allometry["clade"].to_numpy()
    sizes = [
        len(train)
        for train, _ in sa.OrderedGroupSplit(order, min_train_groups=2).split(groups=labels)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_ordered_split_rejects_an_unranked_group(allometry: pd.DataFrame) -> None:
    order = {clade: index for index, clade in enumerate(CLADES[:-1])}
    with pytest.raises(ValueError, match="No ordering value"):
        list(sa.OrderedGroupSplit(order).split(groups=allometry["clade"].to_numpy()))


def test_ordered_split_needs_groups(allometry: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="needs `groups`"):
        list(sa.OrderedGroupSplit({}).split(allometry[FEATURES], allometry["log_rate"]))


def test_extrapolating_along_the_ordering_is_harder_than_leaving_one_out(
    allometry: pd.DataFrame,
) -> None:
    """The point of having both splits: the ordered one must be the harder question.

    Leave-one-group-out surrounds the held-out clade with larger and smaller
    ones. The ordered split does not, and the forest -- which cannot predict
    above its training range at all -- should degrade far more between them.
    """
    labels = allometry["clade"].to_numpy()
    X, y = allometry[FEATURES], allometry["log_rate"].to_numpy()

    lomo = sa.audit_groups(
        X, y, labels, {"forest": RandomForestRegressor(n_estimators=60, random_state=0)}
    )
    lomo_worst = float(lomo["score"].max())

    order = {clade: index for index, clade in enumerate(CLADES)}
    scores = []
    for train_index, test_index in sa.OrderedGroupSplit(order, min_train_groups=2).split(
        groups=labels
    ):
        model = RandomForestRegressor(n_estimators=60, random_state=0)
        model.fit(X.iloc[train_index], y[train_index])
        predicted = model.predict(X.iloc[test_index])
        scores.append(float(np.sqrt(np.mean((y[test_index] - predicted) ** 2))))

    assert max(scores) > lomo_worst
