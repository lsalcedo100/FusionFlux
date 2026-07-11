from __future__ import annotations

import argparse
import json
from pathlib import Path

from features import DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS
from inference import (
    DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
    SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES,
    _default_batch_prediction_output_path,
    predict_batch,
    predict_single_case,
)
from training import train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run FusionFlux.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the ML pipeline.")
    train_parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the input CSV dataset. Required unless --allow-synthetic is set.",
    )
    train_parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate and train on synthetic demo data when --dataset-path is omitted.",
    )
    train_parser.add_argument(
        "--assume-temperature-unit",
        type=str,
        default=None,
        choices=["keV", "eV", "K"],
        help="Explicitly assume this unit for a generic temperature column when temperature_unit is absent.",
    )
    train_parser.add_argument(
        "--shot-prediction-cutoff-rows",
        type=int,
        default=DEFAULT_SHOT_PREDICTION_CUTOFF_ROWS,
        help=(
            "Aggregate each time-resolved shot using measurements available up to this many rows "
            "from shot start."
        ),
    )
    train_parser.add_argument(
        "--skip-report-generation",
        action="store_true",
        help="Skip residual plots and feature-importance explainability artifacts for faster training runs.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict a single fusion operating point.")
    predict_parser.add_argument("--density-m3", type=float, required=True)
    predict_parser.add_argument("--temperature", type=float, required=True)
    predict_parser.add_argument("--temp-unit", type=str, default="keV", choices=["keV", "eV", "K"])
    predict_parser.add_argument("--confinement-time-s", type=float, required=True)
    predict_parser.add_argument("--fuel-purity", type=float, default=None)
    predict_parser.add_argument("--energy-input-mj", type=float, default=None)
    predict_parser.add_argument("--pressure-pa", type=float, default=None)
    predict_parser.add_argument("--ip-ma", type=float, default=None)
    predict_parser.add_argument("--bt-t", type=float, default=None)
    predict_parser.add_argument("--r-m", type=float, default=None)
    predict_parser.add_argument("--a-m", type=float, default=None)
    predict_parser.add_argument("--kappa", type=float, default=None)
    predict_parser.add_argument("--ne-20", type=float, default=None)
    predict_parser.add_argument("--m-amu", type=float, default=None)
    predict_parser.add_argument("--pin-mw", type=float, default=None)
    predict_parser.add_argument("--model-path", type=str, default=None)
    predict_parser.add_argument("--metadata-path", type=str, default=None)
    predict_parser.add_argument("--training-run-id", type=str, default=None)
    predict_parser.add_argument(
        "--default-artifact-selection",
        type=str,
        default=DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
        choices=list(SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES),
        help=(
            "When loading the default saved artifact, prefer either the most runtime-compatible run "
            "or the newest compatible run."
        ),
    )

    predict_batch_parser = subparsers.add_parser(
        "predict-batch",
        help="Predict neutron yield for each prepared operating point or grouped shot in an input CSV.",
    )
    predict_batch_parser.add_argument("--input-csv", type=str, required=True)
    predict_batch_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to write the scored CSV. Defaults to <input_csv_stem>_predictions.csv.",
    )
    predict_batch_parser.add_argument(
        "--assume-temperature-unit",
        type=str,
        default=None,
        choices=["keV", "eV", "K"],
        help="Explicitly assume this unit for a generic temperature column when temperature_unit is absent.",
    )
    predict_batch_parser.add_argument("--model-path", type=str, default=None)
    predict_batch_parser.add_argument("--metadata-path", type=str, default=None)
    predict_batch_parser.add_argument("--training-run-id", type=str, default=None)
    predict_batch_parser.add_argument(
        "--default-artifact-selection",
        type=str,
        default=DEFAULT_ARTIFACT_SELECTION_BEST_COMPATIBILITY,
        choices=list(SUPPORTED_DEFAULT_ARTIFACT_SELECTION_MODES),
        help=(
            "When loading the default saved artifact, prefer either the most runtime-compatible run "
            "or the newest compatible run."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        artifacts = train_models(
            dataset_path=args.dataset_path,
            allow_synthetic=args.allow_synthetic,
            assume_temperature_unit=args.assume_temperature_unit,
            shot_prediction_cutoff_rows=args.shot_prediction_cutoff_rows,
            generate_reports=not args.skip_report_generation,
        )
        print(json.dumps(artifacts, indent=2))
        return

    if args.command == "predict":
        prediction = predict_single_case(
            density_m3=args.density_m3,
            temperature=args.temperature,
            confinement_time_s=args.confinement_time_s,
            temp_unit=args.temp_unit,
            fuel_purity=args.fuel_purity,
            energy_input_mj=args.energy_input_mj,
            pressure_pa=args.pressure_pa,
            ip_ma=args.ip_ma,
            bt_t=args.bt_t,
            r_m=args.r_m,
            a_m=args.a_m,
            kappa=args.kappa,
            ne_20=args.ne_20,
            m_amu=args.m_amu,
            pin_mw=args.pin_mw,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            training_run_id=args.training_run_id,
            default_artifact_selection=args.default_artifact_selection,
        )
        print(json.dumps(prediction, indent=2))
        return

    if args.command == "predict-batch":
        input_path = Path(args.input_csv).expanduser().resolve()
        output_path = (
            Path(args.output_path).expanduser().resolve()
            if args.output_path is not None
            else _default_batch_prediction_output_path(input_path)
        )
        result = predict_batch(
            input_path,
            output_path=output_path,
            assume_temperature_unit=args.assume_temperature_unit,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            training_run_id=args.training_run_id,
            default_artifact_selection=args.default_artifact_selection,
            return_predictions=False,
        )
        print(
            json.dumps(
                {
                    "output_path": str(result.output_path),
                    "row_count": result.row_count,
                    "model_name": result.model_name,
                    "training_run_id": result.training_run_id,
                    "schema_version": result.schema_version,
                    "model_path": str(result.model_path),
                    "metadata_path": str(result.metadata_path),
                    "clipped_negative_prediction_count": result.clipped_negative_prediction_count,
                    "prediction_warnings": result.prediction_warnings,
                    "column_mapping": result.column_mapping,
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
