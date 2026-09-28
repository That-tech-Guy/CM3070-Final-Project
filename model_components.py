"""Reusable model components for the two-stage recommender

The file contains only the preprocessing, supervised models and collaborative
filtering baseline used by ``two_stage_recommender.py``.  Keeping these pieces
separate makes the notebooks easier to read without hiding the calculations
inside one large cell.

Implementation references
-------------------------
Koren, Bell and Volinsky (2009) informed the biased matrix-factorisation
baseline and the use of regularised latent factors
https://doi.org/10.1109/MC.2009.263

The scikit-learn documentation was consulted for ColumnTransformer, Ridge,
RandomForestRegressor and Pipeline
https://scikit-learn.org/stable/modules/compose.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

The matrix-factorisation implementation is deliberately small and transparent
so its cold-start fallback can be inspected directly
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 3070


# This compact implementation keeps the collaborative baseline easy to inspect
class BiasedMatrixFactorization:
    """Explicit-feedback matrix factorisation trained with SGD

    An unseen learner receives the global mean plus any learned resource bias
    This exposes the collaborative model's cold-start limitation instead of
    silently substituting learner metadata
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_epochs: int = 80,
        learning_rate: float = 0.008,
        regularization: float = 0.06,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        user_col: str,
        item_col: str,
        target_col: str,
    ) -> "BiasedMatrixFactorization":
        # Train only on interactions where an outcome was actually observed
        data = frame[[user_col, item_col, target_col]].dropna().copy()
        if data.empty:
            raise ValueError("Matrix factorisation requires at least one observed target")

        # Convert the learner and resource identifiers into stable matrix positions
        self.user_ids_ = sorted(data[user_col].astype(str).unique())
        self.item_ids_ = sorted(data[item_col].astype(str).unique())
        self.user_to_index_ = {
            value: index for index, value in enumerate(self.user_ids_)
        }
        self.item_to_index_ = {
            value: index for index, value in enumerate(self.item_ids_)
        }

        users = data[user_col].astype(str).map(self.user_to_index_).to_numpy(dtype=int)
        items = data[item_col].astype(str).map(self.item_to_index_).to_numpy(dtype=int)
        ratings = data[target_col].to_numpy(dtype=float)

        # Initialise biases at zero and latent factors from a seeded distribution
        rng = np.random.default_rng(self.random_state)
        self.global_mean_ = float(ratings.mean())
        self.rating_min_ = float(ratings.min())
        self.rating_max_ = float(ratings.max())
        self.user_bias_ = np.zeros(len(self.user_ids_), dtype=float)
        self.item_bias_ = np.zeros(len(self.item_ids_), dtype=float)
        self.user_factors_ = rng.normal(
            0.0, 0.08, (len(self.user_ids_), self.n_factors)
        )
        self.item_factors_ = rng.normal(
            0.0, 0.08, (len(self.item_ids_), self.n_factors)
        )

        # A seeded shuffle makes the SGD training order reproducible
        order = np.arange(len(ratings))
        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for row_index in order:
                user_index = users[row_index]
                item_index = items[row_index]
                prediction = (
                    self.global_mean_
                    + self.user_bias_[user_index]
                    + self.item_bias_[item_index]
                    + float(
                        self.user_factors_[user_index]
                        @ self.item_factors_[item_index]
                    )
                )
                error = ratings[row_index] - prediction

                # Update biases and latent factors using the current prediction error
                self.user_bias_[user_index] += self.learning_rate * (
                    error - self.regularization * self.user_bias_[user_index]
                )
                self.item_bias_[item_index] += self.learning_rate * (
                    error - self.regularization * self.item_bias_[item_index]
                )
                previous_user_factors = self.user_factors_[user_index].copy()
                self.user_factors_[user_index] += self.learning_rate * (
                    error * self.item_factors_[item_index]
                    - self.regularization * self.user_factors_[user_index]
                )
                self.item_factors_[item_index] += self.learning_rate * (
                    error * previous_user_factors
                    - self.regularization * self.item_factors_[item_index]
                )
        return self

    def predict(self, users: Iterable[str], items: Iterable[str]) -> np.ndarray:
        # Score each learner-resource pair while handling unseen identifiers explicitly
        predictions: list[float] = []
        for user_value, item_value in zip(users, items):
            user_index = self.user_to_index_.get(str(user_value))
            item_index = self.item_to_index_.get(str(item_value))

            # Known biases remain usable even when one side of the pair is unseen
            prediction = self.global_mean_
            if user_index is not None:
                prediction += self.user_bias_[user_index]
            if item_index is not None:
                prediction += self.item_bias_[item_index]
            if user_index is not None and item_index is not None:
                prediction += float(
                    self.user_factors_[user_index]
                    @ self.item_factors_[item_index]
                )
            predictions.append(
                float(np.clip(prediction, self.rating_min_, self.rating_max_))
            )
        return np.asarray(predictions, dtype=float)

    def save(self, destination: Path) -> None:
        # Store every learned value needed to inspect or reload the fitted baseline
        np.savez_compressed(
            destination,
            global_mean=self.global_mean_,
            rating_min=self.rating_min_,
            rating_max=self.rating_max_,
            user_ids=np.asarray(self.user_ids_, dtype=str),
            item_ids=np.asarray(self.item_ids_, dtype=str),
            user_bias=self.user_bias_,
            item_bias=self.item_bias_,
            user_factors=self.user_factors_,
            item_factors=self.item_factors_,
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            random_state=self.random_state,
        )


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    # Ridge receives scaled numeric values while the tree model keeps original units
    numeric_transformer: object = StandardScaler() if scale_numeric else "passthrough"
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, list(numeric_features)))
    if categorical_features:
        # Unknown categories are ignored so held-out learners do not break prediction
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_ridge_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    alpha: float = 10.0,
) -> Pipeline:
    # Keep preprocessing and regression together so cross-validation stays leakage-safe
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=True,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_random_forest_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    n_estimators: int = 350,
    max_depth: int | None = 12,
    min_samples_leaf: int = 4,
    max_features: str | float = 0.7,
) -> Pipeline:
    # Use one seeded forest so repeated runs produce the same fitted model
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=False,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
