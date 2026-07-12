from __future__ import annotations

import warnings
from contextvars import ContextVar
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
        self._reset_prediction_info_context()
        self.last_prediction_info = _DEFAULT_PREDICTION_INFO

    def _reset_prediction_info_context(self) -> None:
        self._prediction_info_context: ContextVar[ArtifactPredictionInfo] = ContextVar(
            f"fusionflux_last_prediction_info_{id(self)}",
            default=_DEFAULT_PREDICTION_INFO,
        )

    @property
    def last_prediction_info(self) -> ArtifactPredictionInfo:
        return self._prediction_info_context.get()

    @last_prediction_info.setter
    def last_prediction_info(self, value: ArtifactPredictionInfo) -> None:
        self._prediction_info_context.set(value)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_prediction_info_context", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._reset_prediction_info_context()
        self.last_prediction_info = _DEFAULT_PREDICTION_INFO

    def validate_runtime_preprocessing(self) -> tuple[str, ...]:
        from features import (
            LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD,
            assess_runtime_preprocessing_contract_compatibility,
            build_preprocessing_contract,
            describe_preprocessing_contract_differences,
        )

        current_preprocessing = build_preprocessing_contract()
        try:
            legacy_runtime_preprocessing = build_preprocessing_contract(
                fingerprint_method=LEGACY_PREPROCESSING_LOGIC_FINGERPRINT_METHOD,
            )
        except TypeError:
            legacy_runtime_preprocessing = current_preprocessing
        compatibility_report = assess_runtime_preprocessing_contract_compatibility(
            self.fusionflux_preprocessing_contract,
            current_preprocessing,
            legacy_runtime_contract=legacy_runtime_preprocessing,
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

    def predict_with_info(self, features: Any) -> tuple[np.ndarray, ArtifactPredictionInfo]:
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
