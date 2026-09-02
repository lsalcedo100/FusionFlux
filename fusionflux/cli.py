"""The ``fusionflux`` command: the confinement study first, the demo behind it.

``pip install fusionflux`` used to give one console command, and it ran the
synthetic neutron-yield pipeline. That pipeline is careful engineering and it
supports no scientific claim; ``README.md`` says so in its own section, and its
dataset is generated from a hand-written signal. So the single thing the package
offered a new user was the one part of it that measures nothing.

This module makes the front door the study:

    fusionflux predict ...   an energy confinement time, with an interval, an
                             extrapolation distance, and a refusal when the
                             operating point is beyond anything measured here
    fusionflux card          rebuild ``results/predictor.json``, which is what
                             ``predict`` reads
    fusionflux neutron ...   the synthetic pipeline, unchanged, one level down
                             (checkout only: the wheel does not install it)

The ``neutron`` subcommand delegates to ``neutron_yield.fusionflux_cli`` rather
than reimplementing it, so that pipeline's arguments, defaults and behaviour are
defined in exactly one place and cannot drift from the tests that cover them.
Calling that module directly still works and is still tested; this only changes
which command a fresh install puts on the path.
"""

from __future__ import annotations

import argparse
import json
import sys

from fusionflux import predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusionflux",
        description=(
            "Tokamak energy confinement time: predict an operating point, or rebuild "
            "the predictor card. The synthetic neutron-yield demo lives under `neutron`."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict confinement time, with an interval and an out-of-range refusal.",
        description=(
            "Predicts energy confinement time for one operating point. Reports the "
            "extrapolation distance, a calibrated interval, and an explicit warning when "
            "the point sits beyond what this study measured, including when no "
            "range-bounded model can reach the answer at all."
        ),
    )
    for name in predictor.REQUIRED_INPUTS:
        predict_parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=float,
            required=True,
            help=f"{name} (finite and strictly positive)",
        )
    predict_parser.add_argument(
        "--card",
        default=str(predictor.DEFAULT_CARD_PATH),
        help="Predictor card to read (default: results/predictor.json).",
    )
    predict_parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of the readable report."
    )

    card_parser = subparsers.add_parser(
        "card", help="Rebuild the predictor card from the pinned dataset."
    )
    card_parser.add_argument("--output", default=str(predictor.DEFAULT_CARD_PATH))

    # ``REMAINDER`` so the delegated parser owns its own flags entirely: without
    # it, an option this parser happens to define too would be captured here and
    # never reach the pipeline that defines its meaning.
    neutron_parser = subparsers.add_parser(
        "neutron",
        help="The synthetic neutron-yield demo pipeline (supports no scientific claim).",
        description=(
            "Delegates to neutron_yield.fusionflux_cli. This pipeline trains on "
            "synthetic data generated from a hand-written signal, so its accuracy "
            "numbers measure how learnable that generator is and nothing else."
        ),
    )
    neutron_parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments passed through unchanged."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "card":
        print(f"wrote {predictor.save_card(predictor.build_service_card(), args.output)}")
        return

    if args.command == "neutron":
        # Checkout-only, and not as a demotion. `neutron_yield` reaches its data
        # directories through `config.PROJECT_ROOT`, which is derived from that
        # module's own location, so an installed copy would resolve them inside
        # site-packages and write training runs there. The wheel therefore does
        # not ship it, and this says so instead of surfacing a bare import error.
        try:
            from neutron_yield.fusionflux_cli import main as neutron_main
        except ModuleNotFoundError as error:
            raise SystemExit(
                "The neutron-yield pipeline is not installed by the wheel "
                f"({error.name} is missing). It resolves its data directories "
                "relative to its own source file, so an installed copy would read "
                "and write inside site-packages. Clone the repository and run it "
                "from there: see docs/neutron-yield-pipeline.md. Nothing in that "
                "pipeline supports a scientific claim; `fusionflux predict` is the "
                "study and it needs no checkout."
            ) from error

        neutron_main(args.args)
        return

    result = predictor.predict(
        **{name: getattr(args, name) for name in predictor.REQUIRED_INPUTS},
        card=predictor.load_card(args.card),
    )
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        print(predictor.format_prediction(result))


if __name__ == "__main__":
    main(sys.argv[1:])
