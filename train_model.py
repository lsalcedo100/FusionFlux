from __future__ import annotations

from typing import Any

import inference as _inference
import training as _training


def __getattr__(name: str) -> Any:
    if hasattr(_training, name):
        return getattr(_training, name)
    if hasattr(_inference, name):
        return getattr(_inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_training)) | set(dir(_inference)))


def build_parser():
    from fusionflux_cli import build_parser as build_cli_parser

    return build_cli_parser()


def main() -> None:
    from fusionflux_cli import main as cli_main

    cli_main()


__all__ = sorted(set(getattr(_training, "__all__", ())) | set(getattr(_inference, "__all__", ())) | {"build_parser", "main"})


if __name__ == "__main__":
    main()
