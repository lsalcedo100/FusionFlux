"""Install the wheel into a clean interpreter and run the advertised command.

Every other test in this suite runs with the repository root on ``sys.path`` and
``results/`` on disk, which is exactly the condition under which the 0.2.0
packaging defects were invisible. This test removes both: it builds a wheel,
installs it into a fresh virtual environment with no access to the checkout, and
runs the command the README opens with, on ITER's parameters, from a working
directory that is not the repository.

It is the only test here that would have failed on 0.2.0, and it is the reason
`pip install fusionflux` can now be claimed rather than hoped for.

Marked slow: it creates a virtualenv and resolves dependencies from the network.
CI runs it as its own job (`.github/workflows/ci.yml`, `wheel` job).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow

# ITER, the operating point the README prints. Chosen because the interesting
# behaviour is the refusal, and ITER is the case that triggers it.
ITER_ARGS = [
    "--ip-ma", "15",
    "--bt-t", "5.3",
    "--ne-line-1e19-m3", "10",
    "--p-loss-mw", "87",
    "--r-m", "6.2",
    "--inverse-aspect-ratio", "0.3226",
    "--kappa", "1.7",
    "--m-eff-amu", "2.5",
]


@pytest.fixture(scope="module")
def installed_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A venv with only the built wheel installed. Returns the `fusionflux` script."""
    try:
        import build  # noqa: F401
    except ImportError:
        pytest.skip("`build` is not installed; cannot exercise the distribution.")
    if not (PROJECT_ROOT / "results" / "predictor.json").exists():
        pytest.skip("No results/predictor.json to package; run `make results`.")

    # See tests/test_packaging.py: a stale build/lib would be packed too.
    shutil.rmtree(PROJECT_ROOT / "build", ignore_errors=True)

    root = tmp_path_factory.mktemp("smoke")
    dist = root / "dist"
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", str(dist)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"wheel build unavailable here:\n{completed.stderr[-2000:]}")
    (wheel,) = dist.glob("*.whl")

    env_dir = root / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")

    install = subprocess.run(
        [str(bin_dir / "python"), "-m", "pip", "install", "--quiet", str(wheel)],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        pytest.skip(f"cannot install into a clean venv here:\n{install.stderr[-2000:]}")

    script = bin_dir / ("fusionflux.exe" if sys.platform == "win32" else "fusionflux")
    assert script.exists(), "the wheel did not put `fusionflux` on the path"
    return script


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(script), *args], capture_output=True, text=True, cwd=cwd)


def test_predict_runs_outside_the_checkout(installed_cli: Path, tmp_path: Path) -> None:
    """The exact failure 0.2.0 shipped: no results/ directory, so no coefficients."""
    result = _run(installed_cli, "predict", *ITER_ARGS, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_predict_reports_the_refusal_for_iter(installed_cli: Path, tmp_path: Path) -> None:
    """The recommendation is the study's, not a default, and it survives packaging."""
    result = _run(installed_cli, "predict", *ITER_ARGS, "--json", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["physics_exceeds_training_ceiling"] is True
    assert payload["recommended_model"] == "powerlaw_collisionless"
    assert payload["extrapolation_distance"] > 1.0


def test_installed_prediction_matches_the_checkout(installed_cli: Path, tmp_path: Path) -> None:
    """A packaged card that differs from results/ would be a silently wrong tool."""
    from fusionflux import predictor

    result = _run(installed_cli, "predict", *ITER_ARGS, "--json", cwd=tmp_path)
    installed = json.loads(result.stdout)

    local = predictor.predict(
        ip_ma=15.0, bt_t=5.3, ne_line_1e19_m3=10.0, p_loss_mw=87.0,
        r_m=6.2, inverse_aspect_ratio=0.3226, kappa=1.7, m_eff_amu=2.5,
    )
    assert installed["tau_s"] == pytest.approx(local.tau_s, rel=1e-12)


def test_the_wheel_does_not_shadow_a_generic_module(installed_cli: Path, tmp_path: Path) -> None:
    """`import config` in that environment must not resolve to this project."""
    python = installed_cli.parent / "python"
    decoy = tmp_path / "config.py"
    decoy.write_text("MARKER = 'the caller\\'s own config'\n")

    probe = subprocess.run(
        [str(python), "-c", "import config; print(config.MARKER)"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert probe.returncode == 0, (
        "installing fusionflux shadowed the caller's `config` module:\n" + probe.stderr
    )
    assert "the caller's own config" in probe.stdout


def test_python_api_works_from_the_installed_package(installed_cli: Path, tmp_path: Path) -> None:
    """The README's Python snippet, run against the wheel rather than the checkout."""
    python = installed_cli.parent / "python"
    snippet = (
        "from fusionflux import predict\n"
        "r = predict(ip_ma=15.0, bt_t=5.3, ne_line_1e19_m3=10.0, p_loss_mw=87.0,\n"
        "            r_m=6.2, inverse_aspect_ratio=0.3226, kappa=1.7, m_eff_amu=2.5)\n"
        "print(round(r.tau_s, 3), r.physics_exceeds_training_ceiling)\n"
    )
    probe = subprocess.run([str(python), "-c", snippet], capture_output=True, text=True, cwd=tmp_path)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split()[1] == "True"


def test_card_rebuild_explains_it_needs_a_checkout(installed_cli: Path, tmp_path: Path) -> None:
    """`fusionflux card` cannot work from a wheel, and must say why."""
    result = _run(installed_cli, "card", "--output", str(tmp_path / "card.json"), cwd=tmp_path)
    assert result.returncode != 0
    assert "Clone the repository" in (result.stderr + result.stdout)


def test_neutron_explains_it_needs_a_checkout(installed_cli: Path, tmp_path: Path) -> None:
    """Not shipped, so it must name the reason rather than raise an import error."""
    result = _run(installed_cli, "neutron", "train", "--allow-synthetic", cwd=tmp_path)
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "not installed by the wheel" in combined
    assert "ModuleNotFoundError" not in combined.splitlines()[-1]


def test_no_stray_console_scripts(installed_cli: Path) -> None:
    bin_dir = installed_cli.parent
    installed = {path.name for path in bin_dir.iterdir() if path.is_file()}
    # Whatever pip and venv put there, plus ours. Nothing else from this project.
    unexpected = {
        name
        for name in installed
        if name.startswith(("fusion", "neutron", "train", "hdb5"))
    } - {"fusionflux"}
    assert not unexpected, f"unexpected console scripts installed: {sorted(unexpected)}"
