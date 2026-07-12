from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ArtifactPredictionInfo:
    clipped_negative_prediction: bool
    clipped_count: int
    prediction_warnings: tuple[str, ...]


_DEFAULT_PREDICTION_INFO = ArtifactPredictionInfo(
    clipped_negative_prediction=False,
    clipped_count=0,
    prediction_warnings=(),
)


class FusionFluxModelArtifact:
    def __init__(
        self,
        model: Any,
        *,
        schema_version: int,
        training_run_id: str,
        feature_columns: Sequence[str],
        model_name: str,
        preprocessing_contract: Mapping[str, object],
    ) -> None:
        self.model = model
        self.fusionflux_schema_version = schema_version
        self.fusionflux_training_run_id = training_run_id
        self.fusionflux_feature_columns = tuple(feature_columns)
        self.fusionflux_model_name = model_name
        self.fusionflux_preprocessing_contract = dict(preprocessing_contract)
        # Diagnostic record of the most recent predict() call. predict_with_info
        # always overwrites it before anyone reads it, so a plain attribute is
        # sufficient; __setstate__ backfills it for artifacts pickled elsewhere.
        self.last_prediction_info = _DEFAULT_PREDICTION_INFO

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.__dict__.setdefault("last_prediction_info", _DEFAULT_PREDICTION_INFO)

    def validate_runtime_preprocessing(self) -> tuple[str, ...]:
        from features import (
            assess_runtime_preprocessing_contract_compatibility,
            build_preprocessing_contract,
            describe_preprocessing_contract_differences,
        )

        current_preprocessing = build_preprocessing_contract()
        compatibility_report = assess_runtime_preprocessing_contract_compatibility(
            self.fusionflux_preprocessing_contract,
            current_preprocessing,
        )
        if not compatibility_report.compatible:
            differing_fields = list(compatibility_report.differing_fields) or describe_preprocessing_contract_differences(
                self.fusionflux_preprocessing_contract,
                current_preprocessing,
            )
            difference_suffix = f" Changed fields: {', '.join(differing_fields)}." if differing_fields else ""
            raise ValueError(
                "This saved model was trained with a different preprocessing contract than the current runtime code. "
                "Retrain the model or restore the matching preprocessing code before calling predict()."
                f"{difference_suffix}"
            )
        return compatibility_report.warnings

    def predict_with_info(
        self,
        features: Any,
        *,
        revalidate_runtime: bool = True,
    ) -> tuple[np.ndarray, ArtifactPredictionInfo]:
        # ``predict``/direct ``joblib.load(...).predict(...)`` callers revalidate
        # the preprocessing contract on every call so bypassing the inference
        # loader still fails fast on drift. Callers that already validated the
        # contract once (e.g. the batch inference loop that validated at load
        # time) pass ``revalidate_runtime=False`` to skip the repeated, identical
        # fingerprinting work per chunk.
        if revalidate_runtime:
            self.validate_runtime_preprocessing()

        if hasattr(features, "columns"):
            missing_feature_columns = [
                column for column in self.fusionflux_feature_columns if column not in features.columns
            ]
            if missing_feature_columns:
                raise ValueError(
                    "Inference preprocessing did not produce required model features: "
                    f"{missing_feature_columns}"
                )

        predictions = np.asarray(self.model.predict(features), dtype=float)
        negative_mask = predictions < 0
        clipped_count = int(np.count_nonzero(negative_mask))
        prediction_warnings: list[str] = []
        if clipped_count:
            predictions = predictions.copy()
            predictions[negative_mask] = 0.0
            warning_message = "Model predicted a negative neutron yield; output was clipped to 0.0."
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
            prediction_warnings.append(warning_message)

        prediction_info = ArtifactPredictionInfo(
            clipped_negative_prediction=clipped_count > 0,
            clipped_count=clipped_count,
            prediction_warnings=tuple(prediction_warnings),
        )
        self.last_prediction_info = prediction_info
        return predictions, prediction_info

    def predict(self, features: Any) -> np.ndarray:
        predictions, _ = self.predict_with_info(features)
        return predictions
