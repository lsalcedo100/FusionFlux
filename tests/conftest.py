from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def isolated_project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    import config

    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw_dir)

    return {
        "raw": raw_dir,
        "processed": processed_dir,
    }
