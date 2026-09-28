#!/usr/bin/env python3
"""Train and evaluate a two-stage CAPE Unit 1 recommender.

Stage 1 ranks the next topic. Stage 2 ranks external resources inside the
chosen topic. Models train only on selected interactions with observed
simulated score gains. Counterfactual simulator truth is used only after
prediction to evaluate rankings and end-to-end choices.

Implementation references
-------------------------
Koren, Bell and Volinsky (2009) informed the matrix-factorisation baseline
https://doi.org/10.1109/MC.2009.263

The scikit-learn documentation was consulted for grouped cross-validation,
grid search, NDCG and permutation importance
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
https://scikit-learn.org/stable/modules/permutation_importance.html

The SciPy documentation was consulted for the paired Wilcoxon signed-rank test
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from model_components import (
    BiasedMatrixFactorization,
    build_random_forest_pipeline,
    build_ridge_pipeline,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 3070
TARGET = "observed_score_gain"

TOPIC_PERFORMANCE = ["current_score", "weakness_score"]
TOPIC_HISTORY = [
    "sequence_number",
    "simulated_day",
    "prior_interaction_count",
    "prior_completion_rate",
    "prior_mean_observed_gain",
    "topic_prior_interaction_count",
    "topic_prior_completion_rate",
    "topic_prior_mean_observed_gain",
    "topic_seen_before",
    "days_since_topic_interaction",
    "cold_start_flag",
]
TOPIC_GRAPH = [
    "prerequisite_count",
    "prerequisite_min_score",
    "prerequisite_mean_score",
    "unmet_prerequisite_count",
    "prerequisite_ready_60",
    "downstream_mean_weakness",
    "downstream_max_weakness",
    "prerequisite_in_degree",
    "dependent_out_degree",
    "downstream_topic_count",
    "reversed_pagerank",
]
TOPIC_CATALOGUE = [
    "topic_curriculum_order",
    "topic_relative_difficulty",
    "resource_count",
    "resource_format_count",
    "resource_min_difficulty_gap",
    "resource_mean_difficulty_gap",
    "resource_max_tfidf_similarity",
    "resource_mean_tfidf_similarity",
    "resource_mean_duration_minutes",
    "catalogue_format_history_completion",
    "catalogue_format_history_rating",
]
TOPIC_NUMERIC = TOPIC_PERFORMANCE + TOPIC_HISTORY + TOPIC_GRAPH + TOPIC_CATALOGUE
TOPIC_CATEGORICAL = ["candidate_topic_id"]

RESOURCE_PERFORMANCE = ["topic_current_score", "topic_weakness"]
RESOURCE_HISTORY = [
    "sequence_number",
    "simulated_day",
    "prior_interaction_count",
    "prior_completion_rate",
    "prior_mean_observed_gain",
    "topic_prior_interaction_count",
    "topic_prior_completion_rate",
    "topic_prior_mean_observed_gain",
    "format_prior_interaction_count",
    "format_prior_completion_rate",
    "format_prior_mean_rating",
    "cold_start_flag",
]
RESOURCE_CONTENT = [
    "resource_difficulty",
    "resource_duration_minutes",
    "desired_resource_difficulty",
    "difficulty_gap",
    "tfidf_cosine_similarity",
    "topic_curriculum_order",
    "topic_relative_difficulty",
]
RESOURCE_NUMERIC = RESOURCE_PERFORMANCE + RESOURCE_HISTORY + RESOURCE_CONTENT
RESOURCE_CATEGORICAL = ["candidate_topic_id", "resource_format_group"]

PROHIBITED_TOKENS = (
    "oracle",
    "latent",
    "affinity",
    "quality",
    "appeal",
    "mastery",
)


@dataclass
class Bundle:
    topics: pd.DataFrame
    prerequisites: pd.DataFrame
    resources: pd.DataFrame
    topic_features: pd.DataFrame
    resource_features: pd.DataFrame
    topic_oracle: pd.DataFrame
    resource_oracle: pd.DataFrame


def load_bundle(
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    dataset_root: Path | str | None = None,
) -> Bundle:
    project_root = Path(project_root).resolve()
    dataset_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else project_root / "generated_output"
    )
    return Bundle(
        topics=pd.read_csv(dataset_root / "data/topics.csv"),
        prerequisites=pd.read_csv(dataset_root / "data/prerequisites.csv"),
        resources=pd.read_csv(dataset_root / "data/resources.csv"),
        topic_features=pd.read_csv(
            project_root / "two_stage_data/topic_candidates.csv"
        ),
        resource_features=pd.read_csv(
            project_root / "two_stage_data/resource_candidates.csv"
        ),
        topic_oracle=pd.read_csv(
            project_root / "two_stage_internal_truth/topic_candidate_oracle.csv"
        ),
        resource_oracle=pd.read_csv(
            project_root / "two_stage_internal_truth/resource_candidate_oracle.csv"
        ),
    )


def validate_features(numeric: Sequence[str], categorical: Sequence[str]) -> None:
    selected = list(numeric) + list(categorical)
    prohibited = sorted(
        column
        for column in selected
        if any(token in column.lower() for token in PROHIBITED_TOKENS)
    )
    if prohibited:
        raise ValueError(f"Generator truth selected as model features: {prohibited}")


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tune_models(
    training: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> tuple[Any, Any, dict[str, Any]]:
    validate_features(numeric, categorical)
    features = list(numeric) + list(categorical)
    groups = training["learner_id"]
    group_cv = GroupKFold(n_splits=5)

    ridge_search = GridSearchCV(
        build_ridge_pipeline(numeric, categorical),
        {"model__alpha": [1.0, 10.0, 100.0]},
        scoring="neg_mean_absolute_error",
        cv=group_cv,
        n_jobs=1,
        refit=True,
    )
    ridge_search.fit(training[features], training[TARGET], groups=groups)

    forest_search = GridSearchCV(
        build_random_forest_pipeline(numeric, categorical),
        {
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [4, 8],
            "model__max_features": ["sqrt", 0.7],
        },
        scoring="neg_mean_absolute_error",
        cv=group_cv,
        n_jobs=1,
        refit=True,
    )
    forest_search.fit(training[features], training[TARGET], groups=groups)
    tuning = {
        "ridge_best_parameters": ridge_search.best_params_,
        "ridge_grouped_cv_mae": float(-ridge_search.best_score_),
        "random_forest_best_parameters": forest_search.best_params_,
        "random_forest_grouped_cv_mae": float(-forest_search.best_score_),
    }
    return ridge_search.best_estimator_, forest_search.best_estimator_, tuning


def deterministic_random_score(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(
        dtype=np.uint64
    )
    return (hashes ^ np.uint64(RANDOM_SEED)).astype(np.float64) / float(
        np.iinfo(np.uint64).max
    )


def score_topics(candidates: pd.DataFrame, ridge: Any, forest: Any) -> pd.DataFrame:
    scored = candidates.copy()
    scored.loc[:, "Random topic"] = deterministic_random_score(
        scored, ["event_id", "candidate_topic_id"]
    )
    scored.loc[:, "Weakest topic"] = scored["weakness_score"]
    # The large readiness term makes this a lexicographic rule
    # Weakness ranks ready topics and downstream count breaks exact ties
    scored.loc[:, "Prerequisite-ready weakness"] = (
        10.0 * scored["prerequisite_ready_60"]
        + scored["weakness_score"]
        + 0.001 * scored["downstream_topic_count"]
    )
    features = TOPIC_NUMERIC + TOPIC_CATEGORICAL
    scored.loc[:, "Ridge topic"] = ridge.predict(scored[features])
    scored.loc[:, "Random Forest topic"] = forest.predict(scored[features])
    return scored


def score_resources(
    candidates: pd.DataFrame,
    ridge: Any,
    forest: Any,
    matrix_factorization: BiasedMatrixFactorization,
) -> pd.DataFrame:
    scored = candidates.copy()
    scored.loc[:, "Random resource"] = deterministic_random_score(
        scored, ["event_id", "candidate_resource_id"]
    )
    scored.loc[:, "Difficulty match"] = (
        1.0 - np.minimum(scored["difficulty_gap"], 2.0) / 2.0
    )
    scored.loc[:, "TF-IDF content"] = scored["tfidf_cosine_similarity"]
    scored.loc[:, "Matrix factorization"] = matrix_factorization.predict(
        scored["learner_id"], scored["candidate_resource_id"]
    )
    features = RESOURCE_NUMERIC + RESOURCE_CATEGORICAL
    scored.loc[:, "Ridge resource"] = ridge.predict(scored[features])
    scored.loc[:, "Random Forest resource"] = forest.predict(scored[features])
    return scored


def _tie_top1_credit(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_best = int(np.argmax(truth))
    tied = np.flatnonzero(np.isclose(prediction, prediction.max()))
    return float(1.0 / len(tied)) if true_best in tied else 0.0


def _tie_reciprocal_rank(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_best = int(np.argmax(truth))
    target = prediction[true_best]
    above = int(np.sum(prediction > target))
    tied = int(np.sum(np.isclose(prediction, target)))
    return float(1.0 / (1.0 + above + 0.5 * (tied - 1)))


def ranking_metrics(
    scored: pd.DataFrame,
    *,
    group_columns: list[str],
    item_column: str,
    truth_column: str,
    rank_column: str,
    methods: Sequence[str],
    k: int,
    relevant_k: int,
    return_group_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    result_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for method in methods:
        ndcg_gain_values: list[float] = []
        ndcg_rank_values: list[float] = []
        top1_values: list[float] = []
        mrr_values: list[float] = []
        precision_values: list[float] = []
        recall_values: list[float] = []
        regret_values: list[float] = []
        chosen_gain_values: list[float] = []

        for group_key, group in scored.groupby(group_columns, sort=False):
            ordered = group.sort_values(item_column)
            truth = ordered[truth_column].to_numpy(dtype=float)
            oracle_rank = ordered[rank_column].to_numpy(dtype=float)
            prediction = ordered[method].to_numpy(dtype=float)
            rank_relevance = len(group) - oracle_rank
            ndcg_gain = float(ndcg_score([truth], [prediction], k=min(k, len(group))))
            ndcg_rank = float(
                ndcg_score([rank_relevance], [prediction], k=min(k, len(group)))
            )
            top1 = _tie_top1_credit(truth, prediction)
            reciprocal_rank = _tie_reciprocal_rank(truth, prediction)
            true_top = set(np.argsort(-truth, kind="mergesort")[:relevant_k])
            predicted_top = set(
                np.argsort(-prediction, kind="mergesort")[:relevant_k]
            )
            overlap = len(true_top & predicted_top)
            precision = overlap / relevant_k
            recall = overlap / relevant_k
            tied_best = np.flatnonzero(np.isclose(prediction, prediction.max()))
            chosen_gain = float(np.mean(truth[tied_best]))
            denominator = float(truth.max() - truth.min())
            regret = (
                float((truth.max() - chosen_gain) / denominator)
                if denominator > 1e-12
                else 0.0
            )

            ndcg_gain_values.append(ndcg_gain)
            ndcg_rank_values.append(ndcg_rank)
            top1_values.append(top1)
            mrr_values.append(reciprocal_rank)
            precision_values.append(precision)
            recall_values.append(recall)
            regret_values.append(regret)
            chosen_gain_values.append(chosen_gain)
            if return_group_scores:
                key_values = (
                    group_key if isinstance(group_key, tuple) else (group_key,)
                )
                row = dict(zip(group_columns, key_values))
                row.update(
                    {
                        "learner_id": str(ordered["learner_id"].iloc[0]),
                        "method": method,
                        "ndcg_gain": ndcg_gain,
                        "ndcg_rank": ndcg_rank,
                        "normalized_regret": regret,
                    }
                )
                group_rows.append(row)

        result_rows.append(
            {
                "method": method,
                "groups": len(ndcg_gain_values),
                f"ndcg_gain_at_{k}": float(np.mean(ndcg_gain_values)),
                f"ndcg_rank_at_{k}": float(np.mean(ndcg_rank_values)),
                "top1_accuracy": float(np.mean(top1_values)),
                "mrr": float(np.mean(mrr_values)),
                f"precision_at_{relevant_k}": float(np.mean(precision_values)),
                f"recall_at_{relevant_k}": float(np.mean(recall_values)),
                "normalized_regret": float(np.mean(regret_values)),
                "chosen_oracle_gain_points": float(np.mean(chosen_gain_values)),
            }
        )
    metrics = (
        pd.DataFrame(result_rows)
        .sort_values("normalized_regret")
        .reset_index(drop=True)
    )
    return metrics, pd.DataFrame(group_rows) if return_group_scores else None


def selected_training_rows(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    return frame[(frame[flag] == 1) & frame[TARGET].notna()].copy()


def evaluate_prediction_models(
    training: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    ridge: Any,
    forest: Any,
    matrix_factorization: BiasedMatrixFactorization | None = None,
    item_column: str | None = None,
) -> pd.DataFrame:
    predictions: list[tuple[str, np.ndarray]] = [
        ("Global mean", np.repeat(training[TARGET].mean(), len(test))),
        ("Ridge", ridge.predict(test[features])),
        ("Random Forest", forest.predict(test[features])),
    ]
    if matrix_factorization is not None and item_column is not None:
        predictions.append(
            (
                "Matrix factorization",
                matrix_factorization.predict(test["learner_id"], test[item_column]),
            )
        )
    rows = [
        {"method": name, **regression_metrics(test[TARGET], prediction)}
        for name, prediction in predictions
    ]
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def established_evaluation(bundle: Bundle) -> dict[str, Any]:
    topic_selected = selected_training_rows(
        bundle.topic_features, "selected_topic_flag"
    )
    resource_selected = selected_training_rows(
        bundle.resource_features, "selected_resource_flag"
    )
    topic_train = topic_selected[
        topic_selected["chronological_split"] == "train_history"
    ].copy()
    resource_train = resource_selected[
        resource_selected["chronological_split"] == "train_history"
    ].copy()
    established_learners = set(topic_train["learner_id"]) & set(
        resource_train["learner_id"]
    )
    topic_test = topic_selected[
        (topic_selected["chronological_split"] == "future_holdout")
        & topic_selected["learner_id"].isin(established_learners)
    ].copy()
    resource_test = resource_selected[
        (resource_selected["chronological_split"] == "future_holdout")
        & resource_selected["learner_id"].isin(established_learners)
    ].copy()

    topic_ridge, topic_forest, topic_tuning = tune_models(
        topic_train, TOPIC_NUMERIC, TOPIC_CATEGORICAL
    )
    resource_ridge, resource_forest, resource_tuning = tune_models(
        resource_train, RESOURCE_NUMERIC, RESOURCE_CATEGORICAL
    )
    matrix_factorization = BiasedMatrixFactorization().fit(
        resource_train,
        user_col="learner_id",
        item_col="candidate_resource_id",
        target_col=TARGET,
    )

    topic_regression = evaluate_prediction_models(
        topic_train,
        topic_test,
        TOPIC_NUMERIC + TOPIC_CATEGORICAL,
        topic_ridge,
        topic_forest,
    )
    resource_regression = evaluate_prediction_models(
        resource_train,
        resource_test,
        RESOURCE_NUMERIC + RESOURCE_CATEGORICAL,
        resource_ridge,
        resource_forest,
        matrix_factorization,
        "candidate_resource_id",
    )

    # Simulator-only relevance allows ranking evaluation for every future event
    # Regression remains restricted to rows with an observed score-gain target
    future_event_ids = set(
        bundle.topic_features.loc[
            bundle.topic_features["chronological_split"].eq("future_holdout")
            & bundle.topic_features["learner_id"].isin(established_learners),
            "event_id",
        ].unique()
    )
    topic_candidates = bundle.topic_features[
        bundle.topic_features["event_id"].isin(future_event_ids)
    ].merge(
        bundle.topic_oracle,
        on=["event_id", "learner_id", "candidate_topic_id"],
        how="inner",
        validate="one_to_one",
    )
    resource_candidates = bundle.resource_features[
        bundle.resource_features["event_id"].isin(future_event_ids)
    ].merge(
        bundle.resource_oracle,
        on=[
            "event_id",
            "learner_id",
            "candidate_topic_id",
            "candidate_resource_id",
        ],
        how="inner",
        validate="one_to_one",
    )
    scored_topics = score_topics(topic_candidates, topic_ridge, topic_forest)
    scored_resources = score_resources(
        resource_candidates,
        resource_ridge,
        resource_forest,
        matrix_factorization,
    )

    topic_methods = [
        "Random topic",
        "Weakest topic",
        "Prerequisite-ready weakness",
        "Ridge topic",
        "Random Forest topic",
    ]
    topic_ranking, topic_group_scores = ranking_metrics(
        scored_topics,
        group_columns=["event_id"],
        item_column="candidate_topic_id",
        truth_column="oracle_expected_best_resource_gain_points",
        rank_column="oracle_topic_rank",
        methods=topic_methods,
        k=5,
        relevant_k=3,
        return_group_scores=True,
    )

    conditional_resources = scored_resources[
        scored_resources["selected_topic_flag"] == 1
    ].copy()
    resource_methods = [
        "Random resource",
        "Difficulty match",
        "TF-IDF content",
        "Matrix factorization",
        "Ridge resource",
        "Random Forest resource",
    ]
    resource_ranking, resource_group_scores = ranking_metrics(
        conditional_resources,
        group_columns=["event_id", "candidate_topic_id"],
        item_column="candidate_resource_id",
        truth_column="oracle_expected_gain_points",
        rank_column="oracle_resource_rank_within_topic",
        methods=resource_methods,
        k=3,
        relevant_k=2,
        return_group_scores=True,
    )

    return {
        "topic_train": topic_train,
        "topic_test": topic_test,
        "resource_train": resource_train,
        "resource_test": resource_test,
        "topic_ridge": topic_ridge,
        "topic_forest": topic_forest,
        "resource_ridge": resource_ridge,
        "resource_forest": resource_forest,
        "matrix_factorization": matrix_factorization,
        "topic_tuning": topic_tuning,
        "resource_tuning": resource_tuning,
        "topic_regression": topic_regression,
        "resource_regression": resource_regression,
        "topic_ranking": topic_ranking,
        "resource_ranking": resource_ranking,
        "topic_group_scores": topic_group_scores,
        "resource_group_scores": resource_group_scores,
        "scored_topics": scored_topics,
        "scored_resources": scored_resources,
        "established_learners": established_learners,
    }


def choose_highest(
    frame: pd.DataFrame, score_column: str, item_column: str
) -> pd.Series:
    return frame.sort_values(
        [score_column, item_column], ascending=[False, True]
    ).iloc[0]


def end_to_end_evaluation(
    scored_topics: pd.DataFrame,
    scored_resources: pd.DataFrame,
    pipelines: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pipelines is None:
        pipelines = {
            "Random pipeline": ("Random topic", "Random resource"),
            "Rule pipeline": ("Prerequisite-ready weakness", "Difficulty match"),
            "Ridge two-stage": ("Ridge topic", "Ridge resource"),
            "RF topic + difficulty resource": (
                "Random Forest topic",
                "Difficulty match",
            ),
            "RF two-stage": ("Random Forest topic", "Random Forest resource"),
            "Oracle topic + RF resource": (
                "oracle_expected_best_resource_gain_points",
                "Random Forest resource",
            ),
            "Oracle pair": (
                "oracle_expected_best_resource_gain_points",
                "oracle_expected_gain_points",
            ),
        }
    choice_rows: list[dict[str, Any]] = []
    topic_groups = {key: frame for key, frame in scored_topics.groupby("event_id")}
    resource_groups = {
        key: frame for key, frame in scored_resources.groupby("event_id")
    }
    for event_id, topic_group in topic_groups.items():
        resource_group = resource_groups[event_id]
        oracle_best_pair = resource_group.sort_values(
            ["oracle_expected_gain_points", "candidate_resource_id"],
            ascending=[False, True],
        ).iloc[0]
        oracle_worst_gain = float(resource_group["oracle_expected_gain_points"].min())
        oracle_best_gain = float(oracle_best_pair["oracle_expected_gain_points"])
        denominator = oracle_best_gain - oracle_worst_gain
        for pipeline_name, (topic_method, resource_method) in pipelines.items():
            chosen_topic = choose_highest(
                topic_group, topic_method, "candidate_topic_id"
            )
            topic_id = str(chosen_topic["candidate_topic_id"])
            resources_in_topic = resource_group[
                resource_group["candidate_topic_id"] == topic_id
            ]
            chosen_resource = choose_highest(
                resources_in_topic, resource_method, "candidate_resource_id"
            )
            chosen_gain = float(chosen_resource["oracle_expected_gain_points"])
            normalized_regret = (
                (oracle_best_gain - chosen_gain) / denominator
                if denominator > 1e-12
                else 0.0
            )
            choice_rows.append(
                {
                    "event_id": event_id,
                    "learner_id": str(chosen_topic["learner_id"]),
                    "pipeline": pipeline_name,
                    "chosen_topic_id": topic_id,
                    "chosen_resource_id": str(
                        chosen_resource["candidate_resource_id"]
                    ),
                    "chosen_oracle_gain_points": chosen_gain,
                    "oracle_best_gain_points": oracle_best_gain,
                    "normalized_regret": float(normalized_regret),
                    "oracle_topic_hit": int(
                        topic_id == str(oracle_best_pair["candidate_topic_id"])
                    ),
                    "oracle_pair_hit": int(
                        topic_id == str(oracle_best_pair["candidate_topic_id"])
                        and str(chosen_resource["candidate_resource_id"])
                        == str(oracle_best_pair["candidate_resource_id"])
                    ),
                }
            )
    choices = pd.DataFrame(choice_rows)
    summary = (
        choices.groupby("pipeline", as_index=False)
        .agg(
            events=("event_id", "count"),
            chosen_oracle_gain_points=("chosen_oracle_gain_points", "mean"),
            normalized_regret=("normalized_regret", "mean"),
            oracle_topic_hit_rate=("oracle_topic_hit", "mean"),
            oracle_pair_hit_rate=("oracle_pair_hit", "mean"),
        )
        .sort_values("normalized_regret")
        .reset_index(drop=True)
    )
    return summary, choices


def rule_sensitivity(
    scored_topics: pd.DataFrame,
    scored_resources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test reasonable prerequisite and resource-band assumptions.

    Scenarios are declared before comparison and all are retained. They are not
    filtered according to whether they improve the preferred result.
    """

    topics = scored_topics.copy()
    resources = scored_resources.copy()
    prerequisite_thresholds = (50, 60, 70)
    difficulty_bands = {
        # Diagnostic scores occur in 12.5-point steps
        # These values move at least one attainable score between bands
        "35_65": (35.0, 65.0),
        "45_75": (45.0, 75.0),
        "55_85": (55.0, 85.0),
    }
    pipelines: dict[str, tuple[str, str]] = {}
    for threshold in prerequisite_thresholds:
        topic_column = f"rule_topic_prereq_{threshold}"
        ready = topics["prerequisite_min_score"].ge(float(threshold)).astype(int)
        topics.loc[:, topic_column] = (
            10.0 * ready
            + topics["weakness_score"]
            + 0.001 * topics["downstream_topic_count"]
        )
        for band_name, (lower, upper) in difficulty_bands.items():
            resource_column = f"rule_resource_band_{band_name}"
            if resource_column not in resources:
                desired = np.select(
                    [
                        resources["topic_current_score"].lt(lower),
                        resources["topic_current_score"].lt(upper),
                    ],
                    [1, 2],
                    default=3,
                )
                gap = (resources["resource_difficulty"] - desired).abs()
                resources.loc[:, resource_column] = (
                    1.0 - np.minimum(gap, 2.0) / 2.0
                )
            scenario = f"prereq_{threshold}__bands_{band_name}"
            pipelines[scenario] = (topic_column, resource_column)

    summary, choices = end_to_end_evaluation(topics, resources, pipelines)
    default_choices = choices[
        choices["pipeline"] == "prereq_60__bands_45_75"
    ][["event_id", "chosen_topic_id", "chosen_resource_id"]].rename(
        columns={
            "chosen_topic_id": "default_topic_id",
            "chosen_resource_id": "default_resource_id",
        }
    )
    comparison = choices.merge(default_choices, on="event_id", validate="many_to_one")
    stability = (
        comparison.groupby("pipeline", as_index=False)
        .agg(
            same_topic_as_default=(
                "chosen_topic_id",
                lambda values: float(
                    np.mean(
                        values.to_numpy()
                        == comparison.loc[values.index, "default_topic_id"].to_numpy()
                    )
                ),
            ),
            same_resource_as_default=(
                "chosen_resource_id",
                lambda values: float(
                    np.mean(
                        values.to_numpy()
                        == comparison.loc[values.index, "default_resource_id"].to_numpy()
                    )
                ),
            ),
        )
    )
    summary = summary.merge(stability, on="pipeline", validate="one_to_one")
    return summary, choices


def cold_start_evaluation(
    bundle: Bundle,
    seeds: Sequence[int] = (3070, 3071, 3072, 3073, 3074),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate unseen learners with tuning confined to training learners

    Each repeated 80/20 split has its own grouped grid search so no label from a
    held-out learner can influence either fitted parameters or hyperparameters
    """

    learners = np.asarray(sorted(bundle.topic_features["learner_id"].unique()))
    split_topic_rows: list[pd.DataFrame] = []
    split_resource_rows: list[pd.DataFrame] = []
    split_pipeline_rows: list[pd.DataFrame] = []
    split_tuning_rows: list[dict[str, Any]] = []

    for split_number, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        shuffled = learners.copy()
        rng.shuffle(shuffled)
        holdout_size = int(round(0.20 * len(shuffled)))
        test_learners = set(shuffled[:holdout_size])
        train_learners = set(shuffled[holdout_size:])
        print(
            f"  Cold-start split {split_number}/{len(seeds)}: "
            f"tuning on {len(train_learners)} learners",
            flush=True,
        )

        topic_train = selected_training_rows(
            bundle.topic_features[
                bundle.topic_features["learner_id"].isin(train_learners)
            ],
            "selected_topic_flag",
        )
        resource_train = selected_training_rows(
            bundle.resource_features[
                bundle.resource_features["learner_id"].isin(train_learners)
            ],
            "selected_resource_flag",
        )
        topic_ridge, topic_forest, topic_tuning = tune_models(
            topic_train,
            TOPIC_NUMERIC,
            TOPIC_CATEGORICAL,
        )
        resource_ridge, resource_forest, resource_tuning = tune_models(
            resource_train,
            RESOURCE_NUMERIC,
            RESOURCE_CATEGORICAL,
        )
        for stage_name, tuning in [
            ("topic", topic_tuning),
            ("resource", resource_tuning),
        ]:
            split_tuning_rows.append(
                {
                    "split": split_number,
                    "seed": seed,
                    "stage": stage_name,
                    "training_learners": len(train_learners),
                    "held_out_learners": len(test_learners),
                    "ridge_alpha": tuning["ridge_best_parameters"]["model__alpha"],
                    "ridge_grouped_cv_mae": tuning["ridge_grouped_cv_mae"],
                    "forest_max_depth": tuning["random_forest_best_parameters"][
                        "model__max_depth"
                    ],
                    "forest_min_samples_leaf": tuning[
                        "random_forest_best_parameters"
                    ]["model__min_samples_leaf"],
                    "forest_max_features": tuning[
                        "random_forest_best_parameters"
                    ]["model__max_features"],
                    "forest_grouped_cv_mae": tuning[
                        "random_forest_grouped_cv_mae"
                    ],
                }
            )
        matrix_factorization = BiasedMatrixFactorization(random_state=seed).fit(
            resource_train,
            user_col="learner_id",
            item_col="candidate_resource_id",
            target_col=TARGET,
        )

        topic_candidates = bundle.topic_features[
            bundle.topic_features["learner_id"].isin(test_learners)
            & bundle.topic_features["sequence_number"].eq(1)
        ].merge(
            bundle.topic_oracle,
            on=["event_id", "learner_id", "candidate_topic_id"],
            how="inner",
            validate="one_to_one",
        )
        event_ids = set(topic_candidates["event_id"])
        resource_candidates = bundle.resource_features[
            bundle.resource_features["event_id"].isin(event_ids)
        ].merge(
            bundle.resource_oracle,
            on=[
                "event_id",
                "learner_id",
                "candidate_topic_id",
                "candidate_resource_id",
            ],
            how="inner",
            validate="one_to_one",
        )
        scored_topics = score_topics(topic_candidates, topic_ridge, topic_forest)
        scored_resources = score_resources(
            resource_candidates,
            resource_ridge,
            resource_forest,
            matrix_factorization,
        )
        topic_metrics, _ = ranking_metrics(
            scored_topics,
            group_columns=["event_id"],
            item_column="candidate_topic_id",
            truth_column="oracle_expected_best_resource_gain_points",
            rank_column="oracle_topic_rank",
            methods=[
                "Random topic",
                "Weakest topic",
                "Prerequisite-ready weakness",
                "Ridge topic",
                "Random Forest topic",
            ],
            k=5,
            relevant_k=3,
        )
        topic_metrics.insert(0, "split", split_number)
        topic_metrics.insert(1, "seed", seed)
        split_topic_rows.append(topic_metrics)

        conditional_resources = scored_resources[
            scored_resources["selected_topic_flag"] == 1
        ]
        resource_metrics, _ = ranking_metrics(
            conditional_resources,
            group_columns=["event_id", "candidate_topic_id"],
            item_column="candidate_resource_id",
            truth_column="oracle_expected_gain_points",
            rank_column="oracle_resource_rank_within_topic",
            methods=[
                "Random resource",
                "Difficulty match",
                "TF-IDF content",
                "Matrix factorization",
                "Ridge resource",
                "Random Forest resource",
            ],
            k=3,
            relevant_k=2,
        )
        resource_metrics.insert(0, "split", split_number)
        resource_metrics.insert(1, "seed", seed)
        split_resource_rows.append(resource_metrics)

        pipeline_metrics, _ = end_to_end_evaluation(
            scored_topics, scored_resources
        )
        pipeline_metrics.insert(0, "split", split_number)
        pipeline_metrics.insert(1, "seed", seed)
        split_pipeline_rows.append(pipeline_metrics)

    topic_by_split = pd.concat(split_topic_rows, ignore_index=True)
    resource_by_split = pd.concat(split_resource_rows, ignore_index=True)
    pipeline_by_split = pd.concat(split_pipeline_rows, ignore_index=True)

    def summarize(frame: pd.DataFrame, method_column: str) -> pd.DataFrame:
        numeric_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in {"split", "seed"}
        ]
        aggregations: dict[str, tuple[str, str]] = {}
        for column in numeric_columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_sd"] = (column, "std")
        return frame.groupby(method_column, as_index=False).agg(**aggregations)

    topic_summary = summarize(topic_by_split, "method").sort_values(
        "normalized_regret_mean"
    )
    resource_summary = summarize(resource_by_split, "method").sort_values(
        "normalized_regret_mean"
    )
    pipeline_summary = summarize(pipeline_by_split, "pipeline").sort_values(
        "normalized_regret_mean"
    )
    combined_by_split = pd.concat(
        [
            topic_by_split.assign(evaluation="topic_ranking"),
            resource_by_split.assign(evaluation="resource_ranking"),
        ],
        ignore_index=True,
        sort=False,
    )
    return (
        combined_by_split,
        topic_summary,
        resource_summary,
        pipeline_summary,
        pd.DataFrame(split_tuning_rows),
    )


def fixed_model_ablation(
    *,
    stage: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: pd.DataFrame,
    forest_parameters: dict[str, Any],
) -> pd.DataFrame:
    if stage == "topic":
        feature_sets = {
            "Full topic model": (TOPIC_NUMERIC, TOPIC_CATEGORICAL),
            "No graph": (
                TOPIC_PERFORMANCE + TOPIC_HISTORY + TOPIC_CATALOGUE,
                TOPIC_CATEGORICAL,
            ),
            "No history": (
                TOPIC_PERFORMANCE + TOPIC_GRAPH + TOPIC_CATALOGUE,
                TOPIC_CATEGORICAL,
            ),
            "No resource catalogue": (
                TOPIC_PERFORMANCE + TOPIC_HISTORY + TOPIC_GRAPH,
                TOPIC_CATEGORICAL,
            ),
            "Performance only": (TOPIC_PERFORMANCE, []),
        }
        group_columns = ["event_id"]
        item_column = "candidate_topic_id"
        truth_column = "oracle_expected_best_resource_gain_points"
        rank_column = "oracle_topic_rank"
        k, relevant_k = 5, 3
    else:
        feature_sets = {
            "Full resource model": (RESOURCE_NUMERIC, RESOURCE_CATEGORICAL),
            "No interaction history": (
                RESOURCE_PERFORMANCE + RESOURCE_CONTENT,
                RESOURCE_CATEGORICAL,
            ),
            "No resource content": (
                RESOURCE_PERFORMANCE + RESOURCE_HISTORY,
                [],
            ),
            "Performance only": (RESOURCE_PERFORMANCE, []),
        }
        group_columns = ["event_id", "candidate_topic_id"]
        item_column = "candidate_resource_id"
        truth_column = "oracle_expected_gain_points"
        rank_column = "oracle_resource_rank_within_topic"
        k, relevant_k = 3, 2

    rows: list[dict[str, Any]] = []
    for name, (numeric, categorical) in feature_sets.items():
        features = list(numeric) + list(categorical)
        model = build_random_forest_pipeline(
            numeric,
            categorical,
            max_depth=forest_parameters["model__max_depth"],
            min_samples_leaf=int(forest_parameters["model__min_samples_leaf"]),
            max_features=forest_parameters["model__max_features"],
        ).fit(train[features], train[TARGET])
        regression = regression_metrics(test[TARGET], model.predict(test[features]))
        scored = candidates.copy()
        scored[name] = model.predict(scored[features])
        ranking, _ = ranking_metrics(
            scored,
            group_columns=group_columns,
            item_column=item_column,
            truth_column=truth_column,
            rank_column=rank_column,
            methods=[name],
            k=k,
            relevant_k=relevant_k,
        )
        rows.append(
            {
                "feature_set": name,
                **regression,
                **{
                    column: ranking.loc[0, column]
                    for column in ranking.columns
                    if column not in {"method", "groups"}
                },
            }
        )
    return pd.DataFrame(rows).sort_values("normalized_regret").reset_index(drop=True)


def permutation_table(
    model: Any,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        test[features],
        test[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "mae_increase_mean": result.importances_mean,
                "mae_increase_sd": result.importances_std,
            }
        )
        .sort_values("mae_increase_mean", ascending=False)
        .reset_index(drop=True)
    )


def paired_pipeline_test(
    choices: pd.DataFrame,
    candidate: str = "RF two-stage",
    comparator: str = "Rule pipeline",
) -> dict[str, Any]:
    """Run the planned learner-level comparison between RF and the rule baseline"""

    deployable = choices[~choices["pipeline"].str.startswith("Oracle")]
    available = set(deployable["pipeline"])
    required = {candidate, comparator}
    if not required.issubset(available):
        raise ValueError(f"Missing paired-test pipelines: {sorted(required - available)}")
    pivot = deployable.pivot_table(
        index=["learner_id", "event_id"],
        columns="pipeline",
        values="normalized_regret",
    )
    event_difference = pivot[candidate] - pivot[comparator]
    learner_difference = event_difference.groupby(level="learner_id").mean()
    if np.allclose(learner_difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(learner_difference)
    return {
        "comparison": f"{candidate} vs {comparator}",
        "metric": "per-learner mean normalized-regret difference",
        "learners": int(len(learner_difference)),
        "events": int(len(event_difference)),
        "mean_difference": float(learner_difference.mean()),
        "median_difference": float(learner_difference.median()),
        "wilcoxon_statistic": float(statistic),
        "two_sided_p_value": float(p_value),
        "direction": f"Negative values favour {candidate}; positive values favour {comparator}.",
        "claim_boundary": "Controlled simulator evidence only.",
    }


def save_bar_chart(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    destination: Path,
) -> None:
    plot = frame.sort_values(value_column, ascending=False)
    plt.figure(figsize=(9, 4.8))
    plt.barh(plot[label_column], plot[value_column], color="#2F75B5")
    plt.xlabel(value_column.replace("_", " ").title())
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def make_sample_recommendations(
    bundle: Bundle,
    choices: pd.DataFrame,
    scored_topics: pd.DataFrame,
    scored_resources: pd.DataFrame,
) -> pd.DataFrame:
    future_rf = choices[choices["pipeline"] == "RF two-stage"].head(5)
    topic_names = bundle.topics.set_index("topic_id")["topic_name"].to_dict()
    resource_titles = bundle.resources.set_index("resource_id")["title"].to_dict()
    rows: list[dict[str, Any]] = []
    for choice in future_rf.itertuples(index=False):
        topic = scored_topics[
            (scored_topics["event_id"] == choice.event_id)
            & (scored_topics["candidate_topic_id"] == choice.chosen_topic_id)
        ].iloc[0]
        resource = scored_resources[
            (scored_resources["event_id"] == choice.event_id)
            & (scored_resources["candidate_resource_id"] == choice.chosen_resource_id)
        ].iloc[0]
        rows.append(
            {
                "event_id": choice.event_id,
                "learner_id": choice.learner_id,
                "recommended_topic": topic_names[choice.chosen_topic_id],
                "recommended_resource": resource_titles[choice.chosen_resource_id],
                "topic_current_score": float(topic["current_score"]),
                "unmet_prerequisite_count": int(topic["unmet_prerequisite_count"]),
                "resource_difficulty": int(resource["resource_difficulty"]),
                "desired_resource_difficulty": int(
                    resource["desired_resource_difficulty"]
                ),
                "tfidf_similarity": float(resource["tfidf_cosine_similarity"]),
                "controlled_oracle_gain_points": float(
                    choice.chosen_oracle_gain_points
                ),
                "explanation": (
                    f"Topic score {topic['current_score']:.0f}; "
                    f"{int(topic['unmet_prerequisite_count'])} unmet prerequisites; "
                    f"resource difficulty {int(resource['resource_difficulty'])} "
                    f"for desired band {int(resource['desired_resource_difficulty'])}."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_blinded_educator_review(
    bundle: Bundle,
    choices: pd.DataFrame,
    scored_topics: pd.DataFrame,
    scored_resources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create ten deterministic adult-educator comparison cases.

    Eight cases contain different rule/RF recommendations and two contain
    agreements. This selection rule is disclosed and does not use oracle
    performance to choose favourable examples.
    """

    rule = choices[choices["pipeline"] == "Rule pipeline"].copy()
    forest = choices[choices["pipeline"] == "RF two-stage"].copy()
    paired = rule.merge(
        forest,
        on=["event_id", "learner_id"],
        suffixes=("_rule", "_rf"),
        validate="one_to_one",
    ).sort_values("event_id")
    differs = paired[
        paired["chosen_topic_id_rule"].ne(paired["chosen_topic_id_rf"])
        | paired["chosen_resource_id_rule"].ne(paired["chosen_resource_id_rf"])
    ]
    agrees = paired[
        paired["chosen_topic_id_rule"].eq(paired["chosen_topic_id_rf"])
        & paired["chosen_resource_id_rule"].eq(paired["chosen_resource_id_rf"])
    ]

    def spaced_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
        if len(frame) <= count:
            return frame.copy()
        positions = np.linspace(0, len(frame) - 1, count, dtype=int)
        return frame.iloc[positions].copy()

    cases = pd.concat(
        [spaced_sample(differs, 8), spaced_sample(agrees, 2)], ignore_index=True
    ).sort_values("event_id").reset_index(drop=True)
    topic_names = bundle.topics.set_index("topic_id")["topic_name"].to_dict()
    resource_metadata = bundle.resources.set_index("resource_id").to_dict("index")
    prerequisites_by_topic = (
        bundle.prerequisites.groupby("dependent_topic_id")["prerequisite_topic_id"]
        .apply(list)
        .to_dict()
    )
    rng = np.random.default_rng(RANDOM_SEED)
    form_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    def recommendation_details(
        event_id: str, topic_id: str, resource_id: str
    ) -> dict[str, Any]:
        topic = scored_topics[
            scored_topics["event_id"].eq(event_id)
            & scored_topics["candidate_topic_id"].eq(topic_id)
        ].iloc[0]
        resource = scored_resources[
            scored_resources["event_id"].eq(event_id)
            & scored_resources["candidate_resource_id"].eq(resource_id)
        ].iloc[0]
        event_topic_scores = scored_topics[
            scored_topics["event_id"].eq(event_id)
        ].set_index("candidate_topic_id")["current_score"]
        prerequisite_ids = prerequisites_by_topic.get(topic_id, [])
        prerequisite_context = (
            "; ".join(
                f"{topic_names[prerequisite_id]} "
                f"{event_topic_scores.get(prerequisite_id, float('nan')):.0f}"
                for prerequisite_id in prerequisite_ids
            )
            if prerequisite_ids
            else "No direct prerequisite in the provisional graph"
        )
        metadata = resource_metadata[resource_id]
        return {
            "topic": topic_names[topic_id],
            "resource": metadata["title"],
            "provider": metadata["provider"],
            "url": metadata["url"],
            "format": metadata["format_group"],
            "description": metadata["description"],
            "learning_objective": metadata["learning_objective"],
            "prerequisite_context": prerequisite_context,
            "explanation": (
                f"Current topic score {topic['current_score']:.0f}; "
                f"prerequisite evidence: {prerequisite_context}; "
                f"resource difficulty {int(resource['resource_difficulty'])}; "
                f"desired difficulty band {int(resource['desired_resource_difficulty'])}."
            ),
        }

    for index, case in cases.iterrows():
        event_id = str(case["event_id"])
        case_id = f"CASE-{index + 1:02d}"
        event_topics = scored_topics[scored_topics["event_id"].eq(event_id)].merge(
            bundle.topics[["topic_id", "topic_name", "curriculum_order"]],
            left_on="candidate_topic_id",
            right_on="topic_id",
            how="left",
            validate="many_to_one",
        ).sort_values("curriculum_order")
        score_text = "; ".join(
            f"{row.topic_name}: {row.current_score:.0f}"
            for row in event_topics.itertuples(index=False)
        )
        context = event_topics.iloc[0]
        rule_details = recommendation_details(
            event_id,
            str(case["chosen_topic_id_rule"]),
            str(case["chosen_resource_id_rule"]),
        )
        rf_details = recommendation_details(
            event_id,
            str(case["chosen_topic_id_rf"]),
            str(case["chosen_resource_id_rf"]),
        )
        if bool(rng.integers(0, 2)):
            system_a_name, system_a = "Rule pipeline", rule_details
            system_b_name, system_b = "RF two-stage", rf_details
        else:
            system_a_name, system_a = "RF two-stage", rf_details
            system_b_name, system_b = "Rule pipeline", rule_details
        form_rows.append(
            {
                "reviewer_id": "",
                "case_id": case_id,
                "synthetic_learner_label": f"Synthetic learner {index + 1}",
                "topic_scores": score_text,
                "prior_interaction_count": int(context["prior_interaction_count"]),
                "prior_completion_rate": float(context["prior_completion_rate"]),
                "system_a_topic": system_a["topic"],
                "system_a_resource": system_a["resource"],
                "system_a_provider": system_a["provider"],
                "system_a_url": system_a["url"],
                "system_a_format": system_a["format"],
                "system_a_description": system_a["description"],
                "system_a_learning_objective": system_a["learning_objective"],
                "system_a_prerequisite_context": system_a[
                    "prerequisite_context"
                ],
                "system_a_explanation": system_a["explanation"],
                "system_a_curriculum_relevance_1_to_5": "",
                "system_a_sequence_appropriateness_1_to_5": "",
                "system_a_resource_usefulness_1_to_5": "",
                "system_a_explanation_clarity_1_to_5": "",
                "system_b_topic": system_b["topic"],
                "system_b_resource": system_b["resource"],
                "system_b_provider": system_b["provider"],
                "system_b_url": system_b["url"],
                "system_b_format": system_b["format"],
                "system_b_description": system_b["description"],
                "system_b_learning_objective": system_b["learning_objective"],
                "system_b_prerequisite_context": system_b[
                    "prerequisite_context"
                ],
                "system_b_explanation": system_b["explanation"],
                "system_b_curriculum_relevance_1_to_5": "",
                "system_b_sequence_appropriateness_1_to_5": "",
                "system_b_resource_usefulness_1_to_5": "",
                "system_b_explanation_clarity_1_to_5": "",
                "preferred_system_A_B_or_tie": "",
                "reviewer_comments": "",
            }
        )
        key_rows.append(
            {
                "case_id": case_id,
                "event_id": event_id,
                "system_a": system_a_name,
                "system_b": system_b_name,
                "systems_agree": int(
                    rule_details["topic"] == rf_details["topic"]
                    and rule_details["resource"] == rf_details["resource"]
                ),
                "rule_normalized_regret": float(case["normalized_regret_rule"]),
                "rf_normalized_regret": float(case["normalized_regret_rf"]),
                "model_use_allowed": False,
            }
        )
    return pd.DataFrame(form_rows), pd.DataFrame(key_rows)


def run_pipeline(
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    dataset_root: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else root / "generated_output"
    )
    results_dir = root / "two_stage_results"
    figures_dir = results_dir / "figures"
    models_dir = root / "two_stage_saved_models"
    educator_dir = root / "educator_validation"
    internal_truth_dir = root / "two_stage_internal_truth"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    educator_dir.mkdir(exist_ok=True)
    internal_truth_dir.mkdir(exist_ok=True)

    bundle = load_bundle(root, dataset_root)
    print("[1/6] Tuning and evaluating established-user models", flush=True)
    established = established_evaluation(bundle)
    end_to_end, choices = end_to_end_evaluation(
        established["scored_topics"], established["scored_resources"]
    )
    print("[2/6] Running repeated unseen-learner evaluation", flush=True)
    (
        cold_by_split,
        cold_topic,
        cold_resource,
        cold_pipeline,
        cold_tuning,
    ) = cold_start_evaluation(bundle)

    print("[3/6] Running fixed-model ablation", flush=True)
    topic_ablation = fixed_model_ablation(
        stage="topic",
        train=established["topic_train"],
        test=established["topic_test"],
        candidates=established["scored_topics"],
        forest_parameters=established["topic_tuning"][
            "random_forest_best_parameters"
        ],
    )
    resource_ablation_candidates = established["scored_resources"][
        established["scored_resources"]["selected_topic_flag"] == 1
    ]
    resource_ablation = fixed_model_ablation(
        stage="resource",
        train=established["resource_train"],
        test=established["resource_test"],
        candidates=resource_ablation_candidates,
        forest_parameters=established["resource_tuning"][
            "random_forest_best_parameters"
        ],
    )
    print("[4/6] Calculating importance, sensitivity and paired evidence", flush=True)
    topic_importance = permutation_table(
        established["topic_forest"],
        established["topic_test"],
        TOPIC_NUMERIC + TOPIC_CATEGORICAL,
    )
    resource_importance = permutation_table(
        established["resource_forest"],
        established["resource_test"],
        RESOURCE_NUMERIC + RESOURCE_CATEGORICAL,
    )
    paired_test = paired_pipeline_test(choices)
    sensitivity_summary, sensitivity_choices = rule_sensitivity(
        established["scored_topics"], established["scored_resources"]
    )
    educator_review, educator_key = build_blinded_educator_review(
        bundle,
        choices,
        established["scored_topics"],
        established["scored_resources"],
    )
    samples = make_sample_recommendations(
        bundle,
        choices,
        established["scored_topics"],
        established["scored_resources"],
    )

    print("[5/6] Saving result tables and fitted models", flush=True)
    tables = {
        "topic_regression_metrics.csv": established["topic_regression"],
        "resource_regression_metrics.csv": established["resource_regression"],
        "topic_ranking_metrics.csv": established["topic_ranking"],
        "resource_ranking_metrics.csv": established["resource_ranking"],
        "end_to_end_metrics.csv": end_to_end,
        "end_to_end_choices.csv": choices,
        "cold_start_by_split.csv": cold_by_split,
        "cold_start_topic_summary.csv": cold_topic,
        "cold_start_resource_summary.csv": cold_resource,
        "cold_start_pipeline_summary.csv": cold_pipeline,
        "cold_start_tuning.csv": cold_tuning,
        "topic_ablation.csv": topic_ablation,
        "resource_ablation.csv": resource_ablation,
        "topic_permutation_importance.csv": topic_importance,
        "resource_permutation_importance.csv": resource_importance,
        "rule_sensitivity_summary.csv": sensitivity_summary,
        "rule_sensitivity_choices.csv": sensitivity_choices,
        "sample_recommendations.csv": samples,
    }
    for filename, frame in tables.items():
        frame.to_csv(results_dir / filename, index=False, float_format="%.10f")

    tuning = {
        "topic_stage": established["topic_tuning"],
        "resource_stage": established["resource_tuning"],
    }
    (results_dir / "model_tuning.json").write_text(
        json.dumps(tuning, indent=2), encoding="utf-8"
    )
    (results_dir / "paired_pipeline_test.json").write_text(
        json.dumps(paired_test, indent=2), encoding="utf-8"
    )
    educator_review.to_csv(
        educator_dir / "two_stage_blinded_recommendation_review.csv", index=False
    )
    educator_key.to_csv(
        internal_truth_dir / "educator_blinding_key.csv", index=False
    )

    joblib.dump(established["topic_ridge"], models_dir / "topic_ridge.joblib")
    joblib.dump(established["topic_forest"], models_dir / "topic_random_forest.joblib")
    joblib.dump(established["resource_ridge"], models_dir / "resource_ridge.joblib")
    joblib.dump(
        established["resource_forest"], models_dir / "resource_random_forest.joblib"
    )
    established["matrix_factorization"].save(
        models_dir / "resource_matrix_factorization.npz"
    )

    validation = {
        "random_seed": RANDOM_SEED,
        "topic_predictor_count": len(TOPIC_NUMERIC + TOPIC_CATEGORICAL),
        "resource_predictor_count": len(RESOURCE_NUMERIC + RESOURCE_CATEGORICAL),
        "prohibited_topic_predictors": [
            column
            for column in TOPIC_NUMERIC + TOPIC_CATEGORICAL
            if any(token in column.lower() for token in PROHIBITED_TOKENS)
        ],
        "prohibited_resource_predictors": [
            column
            for column in RESOURCE_NUMERIC + RESOURCE_CATEGORICAL
            if any(token in column.lower() for token in PROHIBITED_TOKENS)
        ],
        "topic_training_rows": int(len(established["topic_train"])),
        "topic_test_rows": int(len(established["topic_test"])),
        "resource_training_rows": int(len(established["resource_train"])),
        "resource_test_rows": int(len(established["resource_test"])),
        "established_event_count": int(
            established["scored_topics"]["event_id"].nunique()
        ),
        "ten_topic_candidates_per_event": bool(
            established["scored_topics"].groupby("event_id").size().eq(10).all()
        ),
        "forty_resource_candidates_per_event": bool(
            established["scored_resources"].groupby("event_id").size().eq(40).all()
        ),
        "cold_start_splits": 5,
        "rule_sensitivity_scenarios": int(len(sensitivity_summary)),
        "educator_review_cases": int(len(educator_review)),
        "educator_review_uses_synthetic_profiles_only": True,
        "educator_review_blinding_key_is_separate": True,
        "claim_boundary": "Controlled simulator evidence only; not real learner benefit.",
    }
    (results_dir / "model_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print("[6/6] Saving figures and written summary", flush=True)
    save_bar_chart(
        established["topic_ranking"],
        "method",
        "chosen_oracle_gain_points",
        "Stage 1: established-user topic ranking",
        figures_dir / "topic_ranking.png",
    )
    save_bar_chart(
        established["resource_ranking"],
        "method",
        "chosen_oracle_gain_points",
        "Stage 2: conditional resource ranking",
        figures_dir / "resource_ranking.png",
    )
    save_bar_chart(
        end_to_end,
        "pipeline",
        "chosen_oracle_gain_points",
        "End-to-end two-stage recommendation",
        figures_dir / "end_to_end.png",
    )
    save_bar_chart(
        topic_ablation,
        "feature_set",
        "chosen_oracle_gain_points",
        "Stage 1 Random-Forest ablation",
        figures_dir / "topic_ablation.png",
    )
    save_bar_chart(
        sensitivity_summary,
        "pipeline",
        "chosen_oracle_gain_points",
        "Rule sensitivity across declared assumptions",
        figures_dir / "rule_sensitivity.png",
    )

    best_topic = established["topic_ranking"].iloc[0]
    best_resource = established["resource_ranking"].iloc[0]
    deployable_end_to_end = end_to_end[~end_to_end["pipeline"].str.startswith("Oracle")]
    best_pipeline = deployable_end_to_end.iloc[0]
    alternative_sensitivity = sensitivity_summary[
        sensitivity_summary["pipeline"] != "prereq_60__bands_45_75"
    ]
    summary = [
        "# Two-stage recommender results",
        "",
        "## Controlled findings",
        "",
        f"- Best Stage 1 topic method: **{best_topic['method']}** "
        f"(normalized regret {best_topic['normalized_regret']:.3f}).",
        f"- Best Stage 2 resource method: **{best_resource['method']}** "
        f"(normalized regret {best_resource['normalized_regret']:.3f}).",
        f"- Best deployable end-to-end pipeline: **{best_pipeline['pipeline']}** "
        f"(normalized regret {best_pipeline['normalized_regret']:.3f}).",
        f"- Statistical comparison: {paired_test['comparison']}; mean difference "
        f"{paired_test['mean_difference']:.4f}, p={paired_test['two_sided_p_value']:.4g}.",
        f"- Across the {len(alternative_sensitivity)} alternative rule settings, topic agreement with the "
        f"default ranges from {alternative_sensitivity['same_topic_as_default'].min():.1%} "
        f"to {alternative_sensitivity['same_topic_as_default'].max():.1%}, and full-choice "
        f"agreement ranges from "
        f"{alternative_sensitivity['same_resource_as_default'].min():.1%} to "
        f"{alternative_sensitivity['same_resource_as_default'].max():.1%}.",
        f"- A blinded adult-educator review form was generated with "
        f"{len(educator_review)} synthetic cases; its method key is stored separately.",
        "",
        "## Interpretation boundary",
        "",
        "The redesign makes topic and resource decisions separately and allows graph "
        "features to vary across topic candidates. It still evaluates algorithms against "
        "a disclosed simulator and cannot establish improved learning for real students.",
    ]
    (results_dir / "TWO_STAGE_SUMMARY.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    return {
        "topic_regression": established["topic_regression"],
        "resource_regression": established["resource_regression"],
        "topic_ranking": established["topic_ranking"],
        "resource_ranking": established["resource_ranking"],
        "end_to_end": end_to_end,
        "cold_topic": cold_topic,
        "cold_resource": cold_resource,
        "cold_pipeline": cold_pipeline,
        "cold_tuning": cold_tuning,
        "topic_ablation": topic_ablation,
        "resource_ablation": resource_ablation,
        "sensitivity_summary": sensitivity_summary,
        "educator_review": educator_review,
        "paired_test": paired_test,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Folder containing the model source files",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset output folder, default: <project-root>/generated_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(args.project_root, args.dataset_root)
    print("\nStage 1: established-user topic ranking")
    print(results["topic_ranking"].round(4).to_string(index=False))
    print("\nStage 2: established-user resource ranking")
    print(results["resource_ranking"].round(4).to_string(index=False))
    print("\nEnd-to-end pipelines")
    print(results["end_to_end"].round(4).to_string(index=False))
    print("\nCold-start end-to-end summary")
    print(results["cold_pipeline"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
