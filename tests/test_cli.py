"""Tests for the ``fusionflux`` command, which is what a fresh install puts on the path.

The console script used to run the synthetic neutron-yield pipeline, the one
part of this repository that supports no scientific claim. It now runs the
study, with that pipeline one level down under ``neutron``. Two things have to
hold for that to be a repointing rather than a regression:

* the study's command has to work from the committed card alone, with no
  dataset download, since that is what a fresh install has;
* the delegated pipeline has to keep receiving its arguments unchanged, because
  its behaviour is defined and tested in its own module and this one must not
  acquire a second opinion about it.
"""

from __future__ import annotations

import json

import pytest

from fusionflux import cli, predictor


def _card_or_skip() -> None:
    if not predictor.DEFAULT_CARD_PATH.exists():
        pytest.skip("No predictor card; run `python3 -m fusionflux card`.")


ITER_ARGS = [
    "predict",
    "--ip-ma", "15.0",
    "--bt-t", "5.3",
    "--ne-line-1e19-m3", "10.0",
    "--p-loss-mw", "87.0",
    "--r-m", "6.2",
    "--inverse-aspect-ratio", "0.32258",
    "--kappa", "1.7",
    "--m-eff-amu", "2.5",
]


def test_the_three_subcommands_exist() -> None:
    parser = cli.build_parser()
    for command in ("predict", "card", "neutron"):
        assert parser.parse_args([command] + (ITER_ARGS[1:] if command == "predict" else []))


def test_a_subcommand_is_required() -> None:
    """Bare ``fusionflux`` must not silently do something."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_every_engineering_input_is_required() -> None:
    """Eight inputs, none optional: a defaulted field would predict a different machine."""
    for index in range(1, len(ITER_ARGS), 2):
        partial = ITER_ARGS[:index] + ITER_ARGS[index + 2 :]
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(partial)


def test_predict_prints_a_report_naming_the_refusal(capsys: pytest.CaptureFixture) -> None:
    _card_or_skip()
    cli.main(ITER_ARGS)
    out = capsys.readouterr().out
    assert "power law, collisionless" in out
    assert "cannot exceed" in out
    assert "extrapolation distance" in out


def test_predict_json_is_machine_readable(capsys: pytest.CaptureFixture) -> None:
    _card_or_skip()
    cli.main(ITER_ARGS + ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_model"] == predictor.SAFE_MODEL
    assert payload["physics_exceeds_training_ceiling"] is True
    assert payload["tau_s"] > 0.0


def test_predict_rejects_a_non_positive_input() -> None:
    _card_or_skip()
    bad = list(ITER_ARGS)
    bad[bad.index("--bt-t") + 1] = "0"
    with pytest.raises(ValueError, match="strictly positive"):
        cli.main(bad)


def test_neutron_delegates_its_arguments_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline's own parser must see exactly what was typed after ``neutron``.

    ``REMAINDER`` is what makes that true; without it an option this parser also
    defines would be captured here and never reach the module that defines its
    meaning.
    """
    seen: list[list[str] | None] = []
    import neutron_yield.fusionflux_cli as neutron_cli

    monkeypatch.setattr(neutron_cli, "main", lambda argv=None: seen.append(argv))
    cli.main(["neutron", "train", "--allow-synthetic"])
    assert seen == [["train", "--allow-synthetic"]]


def test_neutron_cli_still_accepts_an_explicit_argv() -> None:
    """The delegation depends on this, and calling it bare must still work."""
    from neutron_yield.fusionflux_cli import build_parser

    args = build_parser().parse_args(["train", "--allow-synthetic"])
    assert args.command == "train"
    assert args.allow_synthetic is True


def test_card_subcommand_targets_the_default_path() -> None:
    args = cli.build_parser().parse_args(["card"])
    assert args.output == str(predictor.DEFAULT_CARD_PATH)


def test_console_script_points_at_this_module() -> None:
    """The packaging claim, checked rather than assumed.

    ``pyproject.toml`` is the only place this is stated, and a stale entry point
    would leave a fresh install running the synthetic demo while every document
    said otherwise.
    """
    import sys
    from pathlib import Path

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # Python 3.10, which `requires-python` still supports
        import tomli as tomllib

    config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    assert config["project"]["scripts"]["fusionflux"] == "fusionflux.cli:main"

    # The entry point has to resolve inside a package the wheel actually ships.
    # It previously named the top-level module `cli`, which was shipped, and that
    # was half the reason installing this project put a dozen generic names into
    # site-packages. tests/test_packaging.py checks the built artifact; this
    # checks the declaration the build reads.
    assert config["tool"]["setuptools"]["packages"] == ["fusionflux"]
    assert "py-modules" not in config["tool"]["setuptools"]
    assert config["tool"]["setuptools"]["package-data"]["fusionflux"] == ["predictor.json"]


def test_module_entry_point_runs_without_an_install() -> None:
    """``python3 -m fusionflux`` is what `make results` calls, so it has to work.

    The card rebuild in the Makefile went through ``python3 predictor.py build``
    while that module was top level. It now lives in the package, and
    ``python3 -m fusionflux.predictor`` reaches it but warns: importing the
    package imports that submodule eagerly, so runpy finds it in ``sys.modules``
    before executing it. ``fusionflux/__main__.py`` exists to give the same entry
    point without the warning, and nothing else would notice if it broke, because
    the console script takes a different path into the same function.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "fusionflux", "predict", "--json",
         "--ip-ma", "15", "--bt-t", "5.3", "--ne-line-1e19-m3", "10",
         "--p-loss-mw", "87", "--r-m", "6.2", "--inverse-aspect-ratio", "0.3226",
         "--kappa", "1.7", "--m-eff-amu", "2.5"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
    assert json.loads(result.stdout)["recommended_model"] == "powerlaw_collisionless"
