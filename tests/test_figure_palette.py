"""One palette for the models, and it has to separate.

Four analysis scripts each carried a copy of the model colours, and the copies
had settled on a booster colour 8.6 from the forest's in OKLab Delta E, where
two series start to read as one below 15, and 0.9 from it under deuteranopia.
Every figure drew the two tree ensembles in what was, to about one reader in
twelve, a single colour.

The palette now lives in ``figures.MODEL_COLORS``. These tests hold two things:
that no script grows its own copy back, and that the colours in the table are
far enough apart. The second is computed here from the OKLab definition rather
than trusted, because a palette that "looks fine" is how this happened.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pytest

import figures

ROOT = Path(__file__).resolve().parent.parent
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

# Two series start to read as one below this, for a reader with full colour vision.
NORMAL_VISION_FLOOR = 15.0
# The colours this palette replaced, which no script may reintroduce.
RETIRED = {
    "#c8873a": "the booster's old colour",
    "#3f8f5c": "the old IPB98(y,2) green",
    "#7d5bbe": "the purple that stood for a split in one panel and for the forest in the next",
}


def _oklab(colour: str) -> np.ndarray:
    """sRGB hex to OKLab, from Ottosson's published matrices."""
    srgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lms = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]) @ linear
    return np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]) @ np.cbrt(lms)


def _delta_e(a: str, b: str) -> float:
    return float(100.0 * np.linalg.norm(_oklab(a) - _oklab(b)))


def test_oklab_matches_its_reference_values() -> None:
    """White is L = 1 with no chroma, and the matrices are easy to transcribe wrongly."""
    assert _oklab("#ffffff") == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)
    assert _oklab("#000000") == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert _oklab("#ff0000") == pytest.approx([0.62796, 0.22486, 0.12585], abs=1e-4)


def test_every_pair_of_model_colours_reads_as_two_colours() -> None:
    pairs = {
        (a, b): _delta_e(figures.MODEL_COLORS[a], figures.MODEL_COLORS[b])
        for a, b in itertools.combinations(figures.MODEL_COLORS, 2)
    }
    too_close = {pair: round(distance, 1) for pair, distance in pairs.items() if distance < NORMAL_VISION_FLOOR}
    assert not too_close, f"these model colours are under {NORMAL_VISION_FLOOR} apart in OKLab Delta E: {too_close}"


def test_the_colours_this_palette_replaced_would_have_failed_the_same_check() -> None:
    """The check has to be able to fail, and it has to fail on the palette that prompted it."""
    assert _delta_e("#eb6834", "#c8873a") < NORMAL_VISION_FLOOR


def test_every_model_with_a_marker_and_a_colour_has_both() -> None:
    assert set(figures.MODEL_COLORS) <= set(figures.MODEL_MARKERS)
    assert set(figures.MODEL_COLORS) <= set(figures.MODEL_LINESTYLES)
    with pytest.raises(KeyError):
        figures.model_color("a_model_nobody_coloured")


def test_the_splits_run_light_to_dark_in_steps_a_reader_can_see() -> None:
    """The splits are an escalation, so their ramp has to be ordered and each step visible."""
    ramp = [figures.SPLIT_RAMP[name] for name in ("grouped_cv", "leave_one_tokamak_out", "size_cut")]
    lightness = [float(_oklab(colour)[0]) for colour in ramp]
    assert lightness == sorted(lightness, reverse=True)
    assert all(_delta_e(a, b) >= NORMAL_VISION_FLOOR for a, b in zip(ramp, ramp[1:], strict=False))


def test_a_context_series_is_dark_enough_to_print() -> None:
    def luminance(colour: str) -> float:
        srgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        return float(linear @ np.array([0.2126, 0.7152, 0.0722]))

    contrast = (luminance("#fcfcfb") + 0.05) / (luminance(figures.CONTEXT_GREY) + 0.05)
    assert contrast >= 3.0, f"{figures.CONTEXT_GREY} is {contrast:.2f}:1 against the page"


def test_no_script_carries_its_own_copy_of_a_model_colour() -> None:
    palette = (
        {colour.lower() for colour in figures.MODEL_COLORS.values()}
        | {colour.lower() for colour in figures.SPLIT_RAMP.values()}
        | {figures.CONTEXT_GREY.lower()}
        | set(RETIRED)
    )
    # The tree-ensemble colours are what drifted. The ridge blue and the forest
    # orange double as generic accent colours in figures that draw no models.
    watched = palette - {figures.MODEL_COLORS["ridge_loglinear"], figures.MODEL_COLORS["random_forest"]}
    offenders = []
    for path in sorted(ROOT.glob("*.py")):
        if path.name == "figures.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for colour in HEX.findall(line):
                if colour.lower() in watched:
                    offenders.append(f"{path.name}:{number} hard-codes {colour}")
    assert not offenders, "use figures.model_color instead:\n  " + "\n  ".join(offenders)
