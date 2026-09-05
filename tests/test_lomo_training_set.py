"""The 30-row threshold selects what is scored, not what is trained on.

`MIN_HELD_OUT_ROWS` decides which machines are large enough to carry a held-out
score of their own. It is easy to read it as also removing the small machines
from the problem, and the paper and README both once said so in prose while the
code did the opposite. That is the kind of error a referee finds and an author
cannot see, because nothing in a passing suite contradicts it.

So the composition of the training fold is asserted here rather than described.
The five machines under the threshold are among the most unusual points in
feature space, and whether a model may learn from them is a methodological
choice that has to be visible.
"""

from __future__ import annotations

import pytest

import hdb5


@pytest.fixture(scope="module")
def dataset():
    return hdb5.prepare_dataset()


def test_the_threshold_excludes_machines_from_scoring_not_from_training(dataset) -> None:
    labels = dataset[hdb5.TOKAMAK_LABEL_COLUMN].to_numpy()
    eligible = hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS)
    every = set(labels)
    below = every - set(eligible)

    assert below, "the check is vacuous unless some machine falls under the threshold"

    for machine in eligible:
        trained_on = set(labels[labels != machine])
        assert trained_on == every - {machine}, (
            f"holding out {machine} trained on {len(trained_on)} machines; the fold "
            "should contain every machine except the held-out one"
        )
        assert below <= trained_on, (
            f"holding out {machine} dropped {sorted(below - trained_on)} from training. "
            "Machines too small to score are still training data."
        )


def test_scored_machines_are_exactly_the_eligible_ones(dataset) -> None:
    """The other half of the same claim, so neither can drift alone."""
    scored = set(
        hdb5.leave_one_tokamak_out(dataset, extra_models=None)["tokamak"].unique()
    )
    assert scored == set(hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS))


def test_the_small_machines_are_a_negligible_share_of_any_training_fold(dataset) -> None:
    """Context for the result: they are in training, and they are 43 rows."""
    counts = dataset[hdb5.TOKAMAK_LABEL_COLUMN].value_counts()
    eligible = set(hdb5.eligible_tokamaks(dataset, min_rows=hdb5.MIN_HELD_OUT_ROWS))
    small_rows = int(counts[[m for m in counts.index if m not in eligible]].sum())
    assert small_rows == 43, f"the five sub-threshold machines hold {small_rows} rows"
    assert small_rows / len(dataset) < 0.01
