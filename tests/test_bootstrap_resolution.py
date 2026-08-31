"""Tests for Result 2b: exponent intervals at three resampling units.

The point of the comparison is that a confidence interval is a statement about a
population, and which population depends entirely on what the bootstrap is
allowed to shuffle. Resampling discharges answers "another shot on these
machines"; the scaling law's actual claim is about other tokamaks, and that
interval is several times wider.

The tests that matter are the ones a plausible refactor could break silently:

* ``test_coarser_units_give_wider_intervals`` is the finding. If someone
  reinstated row-level resampling or pointed the machine level at the wrong
  column, the widening would collapse and nothing else would notice.
* ``test_wall_variants_fold_onto_one_device`` pins the reason the device level
  exists at all. JET and JETILW are one tokamak with two walls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis_scaling_law as asl
import hdb5
from scaling_law import INTERCEPT_NAME


def _multi_machine_dataset(seed: int = 3) -> pd.DataFrame:
    """Machines of unequal size that genuinely disagree about the exponents.

    Two properties have to be built in for this comparison to have anything to
    measure, and both are properties of the real database:

    * unequal sizes, so a few machines dominate the row count, and
    * *machine-level heterogeneity*: each device follows a slightly different
      power law, on top of the within-machine scatter.

    The second is the one that matters. If every machine obeyed one identical
    law and differed only by noise, machines would be exchangeable with each
    other and coarse resampling would be no wider than fine resampling -- the
    comparison would correctly report that there is nothing to widen. Real
    tokamaks are not exchangeable, which is exactly why the coarse interval is
    the honest one, so the fixture reproduces that rather than assuming it.
    """
    rng = np.random.default_rng(seed)
    sizes = {
        "JET": 300,
        "JETILW": 180,
        "AUG": 240,
        "AUGW": 120,
        "ASDEX": 90,
        "D3D": 80,
        "CMOD": 40,
        "NSTX": 60,
        "MAST": 45,
        "JT60U": 55,
    }
    # Exponents are drawn per *device*, so a wall variant inherits its parent
    # machine's law. That is what makes folding the variants back together a
    # coherent operation rather than an averaging-away of real differences.
    device_exponents: dict[str, tuple[float, float]] = {}
    frames = []
    for name, count in sizes.items():
        device = hdb5.WALL_VARIANT_DEVICES.get(name, name)
        if device not in device_exponents:
            device_exponents[device] = (
                0.93 + rng.normal(0.0, 0.15),
                1.97 + rng.normal(0.0, 0.30),
            )
        ip_exponent, r_exponent = device_exponents[device]
        offset = rng.uniform(0.8, 1.25)  # a per-machine systematic, not just noise
        frame = pd.DataFrame(
            {
                "TOK": name,
                "SHOT": rng.integers(0, max(count // 4, 2), count),
                "IP": rng.uniform(0.4, 4.0, count) * offset,
                "BT": rng.uniform(1.0, 5.0, count),
                "NEL": rng.uniform(1.5, 20.0, count),
                "PLTH": rng.uniform(0.5, 25.0, count),
                "RGEO": rng.uniform(0.5, 3.2, count) * offset,
                "KAPPAA": rng.uniform(1.1, 2.2, count),
                "EPS": rng.uniform(0.2, 0.7, count),
                "MEFF": rng.uniform(1.0, 3.0, count),
            }
        )
        frame["ip_exponent"] = ip_exponent
        frame["r_exponent"] = r_exponent
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    clean = (
        0.0562
        * raw["IP"] ** raw["ip_exponent"]
        * raw["BT"] ** 0.15
        * raw["NEL"] ** 0.41
        * raw["PLTH"] ** -0.69
        * raw["RGEO"] ** raw["r_exponent"]
        * raw["EPS"] ** 0.58
        * raw["KAPPAA"] ** 0.78
        * raw["MEFF"] ** 0.19
    )
    raw["TAUTH"] = clean * np.exp(rng.normal(0.0, 0.2, len(raw)))
    return hdb5.prepare_dataset_from_frame(raw.drop(columns=["ip_exponent", "r_exponent"]))


# --- the device mapping -----------------------------------------------------


def test_wall_variants_fold_onto_one_device() -> None:
    """JET and JETILW are one tokamak; ASDEX and AUG are two."""
    dataset = _multi_machine_dataset()
    framed = hdb5.with_device_column(dataset)

    labels = framed[hdb5.TOKAMAK_LABEL_COLUMN]
    devices = framed[hdb5.DEVICE_COLUMN]
    assert set(devices[labels == "JETILW"]) == {"JET"}
    assert set(devices[labels == "AUGW"]) == {"AUG"}
    assert devices.nunique() == labels.nunique() - 2

    # ASDEX Upgrade is a separate machine from ASDEX, not a rewall of it.
    assert "ASDEX" not in hdb5.WALL_VARIANT_DEVICES


def test_device_column_leaves_unmapped_machines_alone() -> None:
    dataset = _multi_machine_dataset()
    framed = hdb5.with_device_column(dataset)
    plain = framed[hdb5.TOKAMAK_LABEL_COLUMN] == "D3D"
    assert set(framed.loc[plain, hdb5.DEVICE_COLUMN]) == {"D3D"}


def test_device_column_does_not_mutate_its_input() -> None:
    dataset = _multi_machine_dataset()
    hdb5.with_device_column(dataset)
    assert hdb5.DEVICE_COLUMN not in dataset.columns


# --- the comparison ---------------------------------------------------------


@pytest.fixture(scope="module")
def resolution() -> asl.BootstrapResolutionComparison:
    dataset = _multi_machine_dataset()
    levels = asl.bootstrap_every_resolution(dataset, n_resamples=200, n_coarse_resamples=200)
    return asl.compare_bootstrap_units(dataset, levels)


def test_every_declared_level_is_actually_run(resolution: asl.BootstrapResolutionComparison) -> None:
    assert resolution.level_names == [name for name, _ in asl.BOOTSTRAP_LEVELS]
    # Nested units, so each level must have strictly fewer resampling units.
    counts = [resolution.units(name) for name in resolution.level_names]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == counts[-2] - 2  # the two wall pairs


# The two exponents ``_multi_machine_dataset`` deliberately varies from machine
# to machine. Everything else in that fixture obeys one identical law.
HETEROGENEOUS = ("ip_ma", "r_m")


def test_coarser_units_widen_exactly_the_heterogeneous_exponents(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    """The finding, stated as the mechanism rather than as a blanket effect.

    Coarse resampling is not wider by arithmetic. Both units draw about the same
    number of rows, so if machines were exchangeable the two intervals would
    coincide, and a test asserting "coarser is always wider" would be asserting
    something untrue. What widens an interval is *between-machine* variation in
    the quantity being estimated, and the fixture puts that variation into two
    exponents and no others.

    So this is the sharp version of the claim: the two exponents that vary by
    machine widen substantially, and the ones that do not vary do not. A bug
    that pointed a level at the wrong column, or that shuffled rows instead of
    whole machines, would fail the first half; a bug that inflated every
    interval mechanically would fail the second.
    """
    for name in resolution.level_names:
        if name == asl.BASELINE_LEVEL:
            continue
        widening = {width.variable: width.widening_factors[name] for width in resolution.widths}
        for variable in HETEROGENEOUS:
            assert widening[variable] > 2.0, (name, variable)
        homogeneous = [
            factor for variable, factor in widening.items() if variable not in HETEROGENEOUS
        ]
        assert max(homogeneous) < 2.0, name


@pytest.mark.parametrize("level", ["machine", "device"])
def test_real_database_widens_on_every_exponent(level: str) -> None:
    """On HDB5 itself, no exponent is exchangeable across machines.

    The synthetic fixture isolates the mechanism; this is the finding as
    reported. Real tokamaks differ in every direction of the design, so every
    interval widens, which is the whole reason the coarse number is the one that
    matches a claim about tokamaks in general.
    """
    if not hdb5.default_hdb5_path().exists():
        pytest.skip("HDB5 STD5 not downloaded; run `python3 hdb5.py download`.")
    dataset = hdb5.prepare_dataset()
    levels = asl.bootstrap_every_resolution(dataset, n_resamples=150, n_coarse_resamples=150)
    resolution = asl.compare_bootstrap_units(dataset, levels)

    assert resolution.median_widening(level) > 2.0
    for width in resolution.widths:
        assert width.widening_factors[level] > 1.5, width.variable
    # And the wider interval must be the more forgiving one about the published
    # exponents: that is the substantive consequence for Result 2's narrative.
    assert resolution.n_published_inside(level) >= resolution.n_published_inside(
        asl.BASELINE_LEVEL
    )


def test_the_baseline_level_is_its_own_reference(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    for width in resolution.widths:
        assert width.widening_factors[asl.BASELINE_LEVEL] == pytest.approx(1.0)
        low, high = width.bounds[asl.BASELINE_LEVEL]
        assert width.width(asl.BASELINE_LEVEL) == pytest.approx(high - low)


def test_a_wider_interval_can_only_admit_more_published_values(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    """Nested intervals should mean nested verdicts, exponent by exponent.

    Not merely a count: an exponent inside the narrow interval must be inside
    the wide one too, unless the intervals are not actually nested. Checking
    this per exponent catches a shifted interval that a total would hide.
    """
    for width in resolution.widths:
        for name in resolution.level_names:
            if name == asl.BASELINE_LEVEL:
                continue
            narrow_low, narrow_high = width.bounds[asl.BASELINE_LEVEL]
            wide_low, wide_high = width.bounds[name]
            if wide_low <= narrow_low and narrow_high <= wide_high:
                assert width.published_inside[name] or not width.published_inside[
                    asl.BASELINE_LEVEL
                ], (width.variable, name)


def test_every_exponent_appears_at_every_level(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    assert len(resolution.widths) == len(asl.IPB98_FEATURE_COLUMNS) + 1  # plus the intercept
    for width in resolution.widths:
        assert set(width.bounds) == set(resolution.level_names)
        assert set(width.widening_factors) == set(resolution.level_names)
        assert set(width.published_inside) == set(resolution.level_names)


def test_the_two_largest_devices_are_reported_with_their_share(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    """The 77% that makes the widening unsurprising, computed rather than quoted."""
    assert len(resolution.largest_two_devices) == 2
    assert resolution.largest_two_devices == ["JET", "AUG"]
    assert 0.0 < resolution.largest_two_row_share < 1.0
    # JET (300 + 180) and AUG (240 + 120) out of 1210 rows.
    assert resolution.largest_two_row_share == pytest.approx(840 / 1210, rel=1e-6)


def test_frame_and_json_agree_on_every_number(
    resolution: asl.BootstrapResolutionComparison,
) -> None:
    frame = resolution.to_frame()
    payload = resolution.to_json()
    assert len(frame) == len(payload["widths"]) == len(resolution.widths)
    assert payload["baseline_level"] == asl.BASELINE_LEVEL
    for name in resolution.level_names:
        assert f"{name}_ci_low" in frame.columns
        assert f"{name}_widening_factor" in frame.columns


# --- degenerate input -------------------------------------------------------


def test_zero_width_baseline_reports_nan_rather_than_an_infinity() -> None:
    """A degenerate baseline must not serialize as inf; ``write_json_strict`` refuses it."""
    dataset = _multi_machine_dataset()
    variables = [INTERCEPT_NAME, *asl.IPB98_FEATURE_COLUMNS]
    degenerate = pd.DataFrame(
        {
            "variable": variables,
            "fitted": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "published_ipb98y2": 0.0,
        }
    )
    wider = degenerate.assign(ci_low=-1.0, ci_high=1.0)
    levels = [
        asl.BootstrapLevel("discharge", hdb5.GROUP_COLUMN, 10, degenerate),
        asl.BootstrapLevel("machine", hdb5.TOKAMAK_LABEL_COLUMN, 6, wider),
        asl.BootstrapLevel("device", hdb5.DEVICE_COLUMN, 4, wider),
    ]
    comparison = asl.compare_bootstrap_units(dataset, levels)
    for width in comparison.widths:
        assert np.isnan(width.widening_factors["machine"])
