#!/usr/bin/env python3
"""Validate the saved two-stage recommender artefacts without retraining.

Use ``--write-hash-baseline`` before a clean rerun and
``--compare-hash-baseline`` afterwards to record byte-level reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
PROHIBITED_TOKENS = ("oracle", "latent", "affinity", "quality", "appeal", "mastery")

# Hash the important generated artefacts so a same-seed rerun can be compared
HASHED_PATHS = [
    "two_stage_data/topic_candidates.csv",
    "two_stage_data/resource_candidates.csv",
    "two_stage_data/two_stage_data_dictionary.csv",
    "two_stage_internal_truth/topic_candidate_oracle.csv",
    "two_stage_internal_truth/resource_candidate_oracle.csv",
    "two_stage_internal_truth/educator_blinding_key.csv",
    "two_stage_validation/candidate_validation.json",
    "two_stage_results/topic_regression_metrics.csv",
    "two_stage_results/resource_regression_metrics.csv",
    "two_stage_results/topic_ranking_metrics.csv",
    "two_stage_results/resource_ranking_metrics.csv",
    "two_stage_results/end_to_end_metrics.csv",
    "two_stage_results/end_to_end_choices.csv",
    "two_stage_results/cold_start_by_split.csv",
    "two_stage_results/cold_start_topic_summary.csv",
    "two_stage_results/cold_start_resource_summary.csv",
    "two_stage_results/cold_start_pipeline_summary.csv",
    "two_stage_results/cold_start_tuning.csv",
    "two_stage_results/topic_ablation.csv",
    "two_stage_results/resource_ablation.csv",
    "two_stage_results/topic_permutation_importance.csv",
    "two_stage_results/resource_permutation_importance.csv",
    "two_stage_results/rule_sensitivity_summary.csv",
    "two_stage_results/rule_sensitivity_choices.csv",
    "two_stage_results/sample_recommendations.csv",
    "two_stage_results/model_tuning.json",
    "two_stage_results/model_validation.json",
    "two_stage_results/paired_pipeline_test.json",
    "two_stage_results/TWO_STAGE_SUMMARY.md",
    "educator_validation/two_stage_blinded_recommendation_review.csv",
]


def sha256(path: Path) -> str:
    # Read large files in blocks instead of loading each file fully into memory
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_hashes(project_root: Path | str = DEFAULT_PROJECT_ROOT) -> dict[str, str]:
    # Missing files indicate an incomplete pipeline run and should fail visibly
    project_root = Path(project_root).resolve()
    missing = [
        relative for relative in HASHED_PATHS if not (project_root / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing reproducibility artefacts: {missing}")
    return {relative: sha256(project_root / relative) for relative in HASHED_PATHS}


def finite_numeric(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    values = numeric.to_numpy(dtype=float)
    # Some combined tables contain NaN where a metric does not apply
    # Reject infinities while allowing those structural missing values
    return bool(np.isfinite(values[~np.isnan(values)]).all())


def validate(
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    dataset_root: Path | str | None = None,
) -> dict[str, bool | int | str]:
    # Load source counts, candidate tables, metrics and separated oracle truth
    project_root = Path(project_root).resolve()
    dataset_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else project_root / "generated_output"
    )
    results = project_root / "two_stage_results"

    interactions = pd.read_csv(dataset_root / "data/interactions.csv")
    topics = pd.read_csv(dataset_root / "data/topics.csv")
    resources = pd.read_csv(dataset_root / "data/resources.csv")
    topic_candidates = pd.read_csv(
        project_root / "two_stage_data/topic_candidates.csv"
    )
    resource_candidates = pd.read_csv(
        project_root / "two_stage_data/resource_candidates.csv"
    )
    topic_oracle = pd.read_csv(
        project_root / "two_stage_internal_truth/topic_candidate_oracle.csv"
    )
    resource_oracle = pd.read_csv(
        project_root / "two_stage_internal_truth/resource_candidate_oracle.csv"
    )
    validation = json.loads((results / "model_validation.json").read_text())
    topic_ranking = pd.read_csv(results / "topic_ranking_metrics.csv")
    resource_ranking = pd.read_csv(results / "resource_ranking_metrics.csv")
    end_to_end = pd.read_csv(results / "end_to_end_metrics.csv")
    cold_by_split = pd.read_csv(results / "cold_start_by_split.csv")
    cold_pipeline = pd.read_csv(results / "cold_start_pipeline_summary.csv")
    cold_tuning = pd.read_csv(results / "cold_start_tuning.csv")
    sensitivity = pd.read_csv(results / "rule_sensitivity_summary.csv")
    # The blank future-review form is checked only for separation and synthetic profiles
    educator_form = pd.read_csv(
        project_root / "educator_validation/two_stage_blinded_recommendation_review.csv"
    )
    educator_key = pd.read_csv(
        project_root / "two_stage_internal_truth/educator_blinding_key.csv"
    )

    # Derive expected row counts from the actual number of events and catalogue items
    event_count = interactions["interaction_id"].nunique()
    expected_topic_rows = event_count * len(topics)
    expected_resource_rows = event_count * len(resources)
    public_columns = list(topic_candidates.columns) + list(resource_candidates.columns)
    prohibited_public = [
        column
        for column in public_columns
        if any(token in column.lower() for token in PROHIBITED_TOKENS)
    ]
    # Reconstruct the established-user test size using the same eligibility rule
    selected_training = topic_candidates[
        topic_candidates["selected_topic_flag"].eq(1)
        & topic_candidates["observed_score_gain"].notna()
        & topic_candidates["chronological_split"].eq("train_history")
    ]
    established_learners = set(selected_training["learner_id"])
    expected_future_events = topic_candidates[
        topic_candidates["chronological_split"].eq("future_holdout")
        & topic_candidates["learner_id"].isin(established_learners)
    ]["event_id"].nunique()
    expected_regression_test_rows = topic_candidates[
        topic_candidates["selected_topic_flag"].eq(1)
        & topic_candidates["observed_score_gain"].notna()
        & topic_candidates["chronological_split"].eq("future_holdout")
        & topic_candidates["learner_id"].isin(established_learners)
    ].shape[0]
    learner_count = interactions["learner_id"].nunique()
    expected_cold_holdout = int(round(0.20 * learner_count))
    expected_cold_training = learner_count - expected_cold_holdout

    # Combine structural, leakage, metric, cold-start and saved-model checks
    checks: dict[str, bool | int | str] = {
        "event_count": int(event_count),
        "topic_candidate_row_count": int(len(topic_candidates)),
        "resource_candidate_row_count": int(len(resource_candidates)),
        "topic_candidate_count_correct": len(topic_candidates) == expected_topic_rows,
        "resource_candidate_count_correct": len(resource_candidates) == expected_resource_rows,
        "ten_topics_per_event": topic_candidates.groupby("event_id").size().eq(10).all(),
        "forty_resources_per_event": resource_candidates.groupby("event_id").size().eq(40).all(),
        "four_resources_per_event_topic": resource_candidates.groupby(
            ["event_id", "candidate_topic_id"]
        ).size().eq(4).all(),
        "one_selected_topic_per_event": topic_candidates.groupby("event_id")[
            "selected_topic_flag"
        ].sum().eq(1).all(),
        "one_selected_resource_per_event": resource_candidates.groupby("event_id")[
            "selected_resource_flag"
        ].sum().eq(1).all(),
        "outcome_only_on_selected_topic": topic_candidates.loc[
            topic_candidates["observed_score_gain"].notna(), "selected_topic_flag"
        ].eq(1).all(),
        "outcome_only_on_selected_resource": resource_candidates.loc[
            resource_candidates["observed_score_gain"].notna(), "selected_resource_flag"
        ].eq(1).all(),
        "no_generator_truth_in_public_candidates": len(prohibited_public) == 0,
        "topic_oracle_is_separate": "oracle_expected_best_resource_gain_points"
        in topic_oracle.columns,
        "resource_oracle_is_separate": "oracle_expected_gain_points"
        in resource_oracle.columns,
        "no_oracle_columns_in_public_educator_form": not any(
            "oracle" in column.lower() or "regret" in column.lower()
            for column in educator_form.columns
        ),
        "all_metric_tables_finite": all(
            finite_numeric(frame)
            for frame in [
                topic_ranking,
                resource_ranking,
                end_to_end,
                cold_by_split,
                cold_pipeline,
                cold_tuning,
                sensitivity,
            ]
        ),
        "established_ranking_event_count_matches_data": topic_ranking["groups"]
        .eq(expected_future_events)
        .all()
        and resource_ranking["groups"].eq(expected_future_events).all(),
        "regression_test_rows_match_observed_targets": validation["topic_test_rows"]
        == expected_regression_test_rows
        and validation["resource_test_rows"] == expected_regression_test_rows,
        "five_cold_start_splits": cold_by_split["split"].nunique() == 5,
        "cold_start_tuning_confined_to_each_training_split": len(cold_tuning) == 10
        and cold_tuning.groupby("split")["stage"].nunique().eq(2).all()
        and cold_tuning["training_learners"].eq(expected_cold_training).all()
        and cold_tuning["held_out_learners"].eq(expected_cold_holdout).all(),
        "nine_sensitivity_scenarios": len(sensitivity) == 9,
        "sensitivity_scenarios_are_non_identical": sensitivity[
            "same_resource_as_default"
        ].nunique()
        > 1,
        # These rows are a blank future protocol rather than completed educator evidence
        "ten_blinded_educator_cases": len(educator_form) == 10
        and len(educator_key) == 10,
        "educator_case_ids_match_key": set(educator_form["case_id"])
        == set(educator_key["case_id"]),
        "educator_profiles_are_synthetic": educator_form[
            "synthetic_learner_label"
        ].str.startswith("Synthetic learner ").all(),
        "educator_key_prohibited_from_modelling": educator_key[
            "model_use_allowed"
        ].eq(False).all(),
        "saved_models_present": all(
            (project_root / "two_stage_saved_models" / filename).exists()
            for filename in [
                "topic_ridge.joblib",
                "topic_random_forest.joblib",
                "resource_ridge.joblib",
                "resource_random_forest.joblib",
                "resource_matrix_factorization.npz",
            ]
        ),
        "claim_boundary": "Controlled simulator evidence only; not real learner benefit.",
    }
    # Collect failed boolean checks while retaining numeric counts for reporting
    failed = [
        name
        for name, value in checks.items()
        if isinstance(value, (bool, np.bool_)) and not bool(value)
    ]
    checks["all_boolean_checks_passed"] = not failed
    checks["failed_checks"] = ", ".join(failed)
    return checks


def main() -> None:
    # Support ordinary validation plus optional before-and-after hash comparison
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Folder containing the model outputs",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset output folder, default: <project-root>/generated_output",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-hash-baseline", action="store_true")
    group.add_argument("--compare-hash-baseline", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results = project_root / "two_stage_results"
    hash_baseline = results / "reproducibility_baseline_hashes.json"
    hash_report = results / "reproducibility_check.json"
    checks = validate(project_root, args.dataset_root)
    if args.write_hash_baseline:
        # Save the first clean run as the reproducibility reference
        hash_baseline.write_text(
            json.dumps(current_hashes(project_root), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote hash baseline: {hash_baseline}")
    elif args.compare_hash_baseline:
        # Compare a later rerun with the saved reference without retraining here
        baseline = json.loads(hash_baseline.read_text(encoding="utf-8"))
        rerun = current_hashes(project_root)
        mismatches = sorted(
            path for path in set(baseline) | set(rerun) if baseline.get(path) != rerun.get(path)
        )
        report = {
            "same_seed_rerun_byte_identical": not mismatches,
            "compared_file_count": len(rerun),
            "mismatched_files": mismatches,
            "baseline_hashes": baseline,
            "rerun_hashes": rerun,
        }
        hash_report.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))

    # Print the complete audit and return a non-zero exit code if any check failed
    print(json.dumps(checks, indent=2, default=bool))
    if not checks["all_boolean_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
