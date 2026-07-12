from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    ensure_parent_directory(path)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temp_path
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    with atomic_output_path(path) as temp_path:
        temp_path.write_text(text, encoding=encoding)


def _json_safe(value: object) -> object:
    # JSON has no representation for NaN/Infinity, and we serialize with
    # allow_nan=False so that readers never have to parse non-standard tokens.
    # Metadata can legitimately carry non-finite floats (e.g. an r2 or high-yield
    # threshold that is undefined for a given run), so map those to null rather
    # than letting json.dumps raise and abort an otherwise-complete training run.
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_strict(path: Path, payload: dict[str, object]) -> None:
    write_text_atomic(
        path,
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
    )


def write_dataframe_csv_atomic(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    with atomic_output_path(path) as temp_path:
        frame.to_csv(temp_path, index=index)
