"""Model factory registry for the FusionFlux training pipeline.

Defines the preprocessing transformer and the family of candidate regressors that
``training.train_models`` cross-validates and selects among. Kept separate from
the training orchestration so the model definitions can evolve independently.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE

ModelFactory = Callable[[], TransformedTargetRegressor]

# The median DummyRegressor is a reference floor, not a shippable model. On an
# exact metric tie the alphabetical "model" tie-break would rank "baseline" first
# and ship it, so training.train_models demotes it to last and it can only win by
# being strictly better than every real model family.
BASELINE_MODEL_NAME = "baseline"


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_columns,
            )
        ]
    )


def build_model_registry(feature_columns: list[str]) -> dict[str, ModelFactory]:
    return {
        BASELINE_MODEL_NAME: lambda: TransformedTargetRegressor(
            regressor=Pipeline([("prep", build_preprocessor(feature_columns)), ("model", DummyRegressor(strategy="median"))]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "random_forest": lambda: TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("prep", build_preprocessor(feature_columns)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=14,
                            min_samples_leaf=2,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "hist_gradient_boosting": lambda: TransformedTargetRegressor(
            regressor=Pipeline(
                [
                    ("prep", build_preprocessor(feature_columns)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_depth=8,
                            learning_rate=0.05,
                            max_iter=350,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }
