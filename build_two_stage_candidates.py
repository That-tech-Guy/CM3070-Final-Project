#!/usr/bin/env python3
"""Build leakage-safe candidate tables for a two-stage CAPE recommender



Stage 1 compares all  ten (reasearcher choosen) topics at each historical decision point. Stage 2
compares the four external resources assigned to a candidate topic. Public
feature tables contain only information available before the event. Simulator
oracle values are written separately and are reserved for evaluation
"""

from __future__ import annotations

import argparse

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_dataset import desired_resource_difficulty, sigmoid


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent



# this is a safe mean that ignores NaN values and returns a default if the list is empty
def safe_mean(values: list[float], default: float = 0.0) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else float(default)


# this function builds a dictionary of all downstream topics for each topic in the curriculum graph
def descendants_by_topic(
    topic_ids: list[str], prerequisites: pd.DataFrame
) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {topic_id: [] for topic_id in topic_ids}
    for source, target in prerequisites[
        ["prerequisite_topic_id", "dependent_topic_id"]
    ].itertuples(index=False, name=None):
        adjacency[source].append(target)

    descendants: dict[str, list[str]] = {}
    for topic_id in topic_ids:
        found: set[str] = set()
        stack = list(adjacency[topic_id])
        while stack:
            node = stack.pop()
            if node not in found:
                found.add(node)
                stack.extend(adjacency[node])
        descendants[topic_id] = sorted(found)
    return descendants

    # This function summarizes the history of interactions for a learner, 
    # optionally filtered by topic and format group. It calculates the count of interactions, 
    # completion rate, mean observed score gain, and mean usefulness rating
def history_summary(
    history: list[dict[str, Any]],
    *,
    topic_id: str | None = None,
    format_group: str | None = None,
) -> dict[str, float | int]:
    selected = history
    if topic_id is not None:
        selected = [row for row in selected if row["target_topic_id"] == topic_id]
    if format_group is not None:
        selected = [row for row in selected if row["format_group"] == format_group]
    gains = [row["score_gain"] for row in selected if pd.notna(row["score_gain"])]
    ratings = [
        row["usefulness_rating"]
        for row in selected
        if pd.notna(row["usefulness_rating"])
    ]
    return {
        "count": len(selected),
        "completion_rate": safe_mean([row["completed"] for row in selected]),
        "mean_observed_gain": safe_mean(gains),
        "mean_rating": safe_mean(ratings),
    }


# here we rebuild the expected outcome for a given resource, using the same logic as the simulator
def expected_resource_outcome(
    *,
    config: dict[str, Any],
    learner_truth: dict[str, Any],
    resource_truth: dict[str, float],
    resource: Any,
    desired_difficulty: int,
    topic_mastery: float,
    prerequisite_readiness: float,
) -> tuple[float, float, float, float]:
    # Rebuild the simulator-only expected outcome for evaluation

    #TF-IDF affected the logged selection policy in dataset creation but was not
    #defined as a direct cause of learning gain, so it is not used in this oracle 
  

    difficulty_match = 1.0 - abs(int(resource.difficulty) - desired_difficulty) / 2.0
    format_match = float(learner_truth[f"affinity_{resource.format_group}"])
    duration_scaled = (float(resource.duration_minutes) - 10.0) / 70.0
    overchallenge = max(0.0, int(resource.difficulty) - desired_difficulty) / 2.0

    completion = config["completion"]
    completion_logit = (
        float(completion["intercept"])
        + float(completion["engagement_weight"])
        * float(learner_truth["engagement_propensity"])
        + float(completion["format_match_weight"]) * format_match
        + float(completion["difficulty_match_weight"]) * difficulty_match
        + float(completion["latent_appeal_weight"])
        * float(resource_truth["latent_appeal"])
        - float(completion["duration_penalty"]) * duration_scaled
        - float(completion["overchallenge_penalty"]) * overchallenge
    )
    completion_probability = float(sigmoid(completion_logit))
    # the expected mastery gain is a function of the latent quality of the resource, 
    # the match to the learner's preferred format, the match to the desired difficulty, and the readiness of 
    # the learner's prerequisites. The final expected mastery gain is scaled by the probability of completion 
    # and an incomplete exposure multiplier
    learning = config["latent_learning"]
    complete_effect = (
        float(learning["base"])
        + float(learning["resource_quality_weight"])
        * float(resource_truth["latent_quality"])
        + float(learning["format_match_weight"]) * format_match
        + float(learning["difficulty_match_weight"]) * difficulty_match
        + float(learning["prerequisite_readiness_weight"])
        * prerequisite_readiness
    )
    full_mastery_gain = (
        complete_effect
        * float(learner_truth["learning_responsiveness"])
        * max(0.02, 1.0 - topic_mastery)
    )
    incomplete_multiplier = float(learning["incomplete_exposure_multiplier"])
    expected_mastery_gain = full_mastery_gain * (
        completion_probability
        + (1.0 - completion_probability) * incomplete_multiplier
    )
    expected_mastery_gain = float(
        np.clip(
            expected_mastery_gain,
            0.0,
            float(learning["maximum_single_interaction_mastery_gain"]),
        )
    )
    return (
        completion_probability,
        expected_mastery_gain,
        difficulty_match,
        format_match,
    )

# for each historical event, we reconstruct the candidate tables for all topics and resources,
# using only information available before the event. We also compute the simulator oracle values for evaluation
def build_candidate_tables(
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    dataset_root: Path | str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    #Reconstruct every topic and resource choice using pre-event information


    project_root = Path(project_root).resolve()
    dataset_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else project_root / "generated_output"
    )
    config = json.loads((project_root / "config.json").read_text(encoding="utf-8"))
    topics = pd.read_csv(dataset_root / "data/topics.csv")
    prerequisites = pd.read_csv(dataset_root / "data/prerequisites.csv")
    resources = pd.read_csv(dataset_root / "data/resources.csv")
    similarities = pd.read_csv(dataset_root / "data/resource_topic_similarity.csv")
    interactions = pd.read_csv(dataset_root / "data/interactions.csv")
    baselines = pd.read_csv(dataset_root / "data/learner_topic_baseline.csv")
    learner_topic_truth = pd.read_csv(
        dataset_root / "internal_generator_truth/learner_topic_truth.csv"
    )
    learner_truth = pd.read_csv(
        dataset_root / "internal_generator_truth/learner_truth.csv"
    )
    resource_truth = pd.read_csv(
        dataset_root / "internal_generator_truth/resource_truth.csv"
    )
    selected_truth = pd.read_csv(
        dataset_root / "internal_generator_truth/selected_interaction_truth.csv"
    )
    original_resource_oracle = pd.read_csv(
        dataset_root / "internal_generator_truth/candidate_potential_outcomes.csv"
    )

    topic_ids = topics["topic_id"].astype(str).tolist()
    topic_lookup = topics.set_index("topic_id").to_dict("index")
    prerequisite_map: dict[str, list[str]] = defaultdict(list)
    for prerequisite, dependent in prerequisites[
        ["prerequisite_topic_id", "dependent_topic_id"]
    ].itertuples(index=False, name=None):
        prerequisite_map[str(dependent)].append(str(prerequisite))
    descendants = descendants_by_topic(topic_ids, prerequisites)
    resources_by_topic = {
        str(topic_id): frame.reset_index(drop=True)
        for topic_id, frame in resources.groupby("primary_topic_id", sort=False)
    }
    similarity_lookup = {
        (str(row.resource_id), str(row.topic_id)): float(row.tfidf_cosine_similarity)
        for row in similarities.itertuples(index=False)
    }
    resource_truth_lookup = {
        str(resource_id): {
            "latent_quality": float(latent_quality),
            "latent_appeal": float(latent_appeal),
        }
        for resource_id, latent_quality, latent_appeal in resource_truth[
            ["resource_id", "latent_quality", "latent_appeal"]
        ].itertuples(index=False, name=None)
    }
    learner_truth_lookup = learner_truth.set_index("learner_id").to_dict("index")

    events = interactions.merge(
        selected_truth,
        on="interaction_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_truth"),
    ).sort_values(["learner_id", "sequence_number"])

    baseline_score_lookup = {
        (str(row.learner_id), str(row.topic_id)): float(row.diagnostic_score)
        for row in baselines.itertuples(index=False)
    }
    mastery_lookup = {
        (str(row.learner_id), str(row.topic_id)): float(row.initial_latent_mastery)
        for row in learner_topic_truth.itertuples(index=False)
    }
    # store the reconstructed candidate features and oracle values for each event
    topic_feature_rows: list[dict[str, Any]] = []
    resource_feature_rows: list[dict[str, Any]] = []
    topic_oracle_rows: list[dict[str, Any]] = []
    resource_oracle_rows: list[dict[str, Any]] = []
    selected_score_differences: list[float] = []
    mastery_state_differences: list[float] = []

    for learner_id, learner_events in events.groupby("learner_id", sort=False):
        learner_id = str(learner_id)
        current_scores = {
            topic_id: baseline_score_lookup[(learner_id, topic_id)]
            for topic_id in topic_ids
        }
        current_mastery = {
            topic_id: mastery_lookup[(learner_id, topic_id)]
            for topic_id in topic_ids
        }
        history: list[dict[str, Any]] = []
        last_topic_day: dict[str, int] = {}
        hidden_learner = learner_truth_lookup[learner_id]

        for event in learner_events.itertuples(index=False):
            event_id = str(event.event_id)
            selected_topic = str(event.target_topic_id)
            selected_resource = str(event.resource_id)
            selected_score_differences.append(
                abs(current_scores[selected_topic] - float(event.pre_score))
            )
            mastery_state_differences.append(
                abs(current_mastery[selected_topic] - float(event.latent_mastery_before))
            )
            # reconstruct candidate features and oracle values for the selected topic and resource
            overall = history_summary(history)
            all_topic_oracles: list[dict[str, Any]] = []
            for topic_id in topic_ids:
                topic = topic_lookup[topic_id]
                current_score = float(current_scores[topic_id])
                weakness = 1.0 - current_score / 100.0
                desired_difficulty = desired_resource_difficulty(current_score)
                topic_history = history_summary(history, topic_id=topic_id)
                prerequisite_ids = prerequisite_map[topic_id]
                prerequisite_scores = [current_scores[value] for value in prerequisite_ids]
                prerequisite_mastery = [current_mastery[value] for value in prerequisite_ids]
                prerequisite_min_score = (
                    float(min(prerequisite_scores)) if prerequisite_scores else 100.0
                )
                prerequisite_mean_score = safe_mean(prerequisite_scores, 100.0)
                prerequisite_readiness = safe_mean(prerequisite_mastery, 1.0)
                unmet_prerequisite_count = int(
                    sum(value < 60.0 for value in prerequisite_scores)
                )
                downstream_weaknesses = [
                    1.0 - current_scores[value] / 100.0
                    for value in descendants[topic_id]
                ]
                downstream_mean_weakness = safe_mean(downstream_weaknesses)
                downstream_max_weakness = (
                    float(max(downstream_weaknesses))
                    if downstream_weaknesses
                    else 0.0
                )
                seen_before = int(topic_id in last_topic_day)
                days_since_topic = (
                    int(event.simulated_day) - last_topic_day[topic_id]
                    if seen_before
                    else 0
                )

                candidate_resources = resources_by_topic[topic_id]
                difficulty_gaps: list[float] = []
                similarities_for_topic: list[float] = []
                historical_format_completion: list[float] = []
                historical_format_rating: list[float] = []
                topic_resource_oracles: list[dict[str, Any]] = []

                for resource in candidate_resources.itertuples(index=False):
                    resource_id = str(resource.resource_id)
                    semantic_similarity = similarity_lookup[(resource_id, topic_id)]
                    format_history = history_summary(
                        history, format_group=str(resource.format_group)
                    )
                    (
                        completion_probability,
                        expected_mastery_gain,
                        difficulty_match,
                        _,
                    ) = expected_resource_outcome(
                        config=config,
                        learner_truth=hidden_learner,
                        resource_truth=resource_truth_lookup[resource_id],
                        resource=resource,
                        desired_difficulty=desired_difficulty,
                        topic_mastery=current_mastery[topic_id],
                        prerequisite_readiness=prerequisite_readiness,
                    )
                    difficulty_gap = abs(int(resource.difficulty) - desired_difficulty)
                    difficulty_gaps.append(float(difficulty_gap))
                    similarities_for_topic.append(float(semantic_similarity))
                    historical_format_completion.append(
                        float(format_history["completion_rate"])
                    )
                    # the historical format rating is used as a proxy for latent appeal, which is 
                    # not observable in the public dataset
                    historical_format_rating.append(float(format_history["mean_rating"]))

                    selected_topic_flag = int(topic_id == selected_topic)
                    selected_resource_flag = int(
                        selected_topic_flag and resource_id == selected_resource
                    )
                    # the observed gain is only recorded for the selected resource, and is NaN for all other candidates
                    observed_gain = (
                        float(event.score_gain)
                        if selected_resource_flag and pd.notna(event.score_gain)
                        else np.nan
                    )
                    resource_feature_rows.append(
                        {
                            "event_id": event_id,
                            "interaction_id": str(event.interaction_id),
                            "learner_id": learner_id,
                            "sequence_number": int(event.sequence_number),
                            "simulated_day": int(event.simulated_day),
                            "candidate_topic_id": topic_id,
                            "candidate_resource_id": resource_id,
                            "selected_topic_flag": selected_topic_flag,
                            "selected_resource_flag": selected_resource_flag,
                            "chronological_split": str(event.chronological_split),
                            "topic_current_score": current_score,
                            "topic_weakness": weakness,
                            "prior_interaction_count": int(overall["count"]),
                            "prior_completion_rate": float(overall["completion_rate"]),
                            "prior_mean_observed_gain": float(
                                overall["mean_observed_gain"]
                            ),
                            "topic_prior_interaction_count": int(topic_history["count"]),
                            "topic_prior_completion_rate": float(
                                topic_history["completion_rate"]
                            ),
                            "topic_prior_mean_observed_gain": float(
                                topic_history["mean_observed_gain"]
                            ),
                            "format_prior_interaction_count": int(
                                format_history["count"]
                            ),
                            "format_prior_completion_rate": float(
                                format_history["completion_rate"]
                            ),
                            "format_prior_mean_rating": float(
                                format_history["mean_rating"]
                            ),
                            "cold_start_flag": int(len(history) < 3),
                            "resource_format_group": str(resource.format_group),
                            "resource_difficulty": int(resource.difficulty),
                            "resource_duration_minutes": float(
                                resource.duration_minutes
                            ),
                            "desired_resource_difficulty": int(desired_difficulty),
                            "difficulty_gap": float(difficulty_gap),
                            "tfidf_cosine_similarity": float(semantic_similarity),
                            "topic_curriculum_order": int(topic["curriculum_order"]),
                            "topic_relative_difficulty": int(
                                topic["relative_difficulty"]
                            ),
                            "observed_score_gain": observed_gain,
                        }
                    )
                    # the oracle values are computed using the simulator logic and are not observable in the public dataset
                    oracle_row = {
                        "event_id": event_id,
                        "learner_id": learner_id,
                        "candidate_topic_id": topic_id,
                        "candidate_resource_id": resource_id,
                        "oracle_completion_probability": completion_probability,
                        "oracle_expected_mastery_gain": expected_mastery_gain,
                        "oracle_expected_gain_points": 100.0 * expected_mastery_gain,
                        "model_use_allowed": False,
                    }
                    resource_oracle_rows.append(oracle_row)
                    topic_resource_oracles.append(oracle_row)

                best_resource = max(
                    topic_resource_oracles,
                    key=lambda row: row["oracle_expected_mastery_gain"],
                )
                # the topic oracle values are computed using the best resource for each topic, and ae 
                # not observable in the public dataset
                topic_oracle = {
                    "event_id": event_id,
                    "learner_id": learner_id,
                    "candidate_topic_id": topic_id,
                    "oracle_best_resource_id": best_resource[
                        "candidate_resource_id"
                    ],
                    "oracle_expected_best_resource_gain": float(
                        best_resource["oracle_expected_mastery_gain"]
                    ),
                    "oracle_expected_best_resource_gain_points": float(
                        best_resource["oracle_expected_gain_points"]
                    ),
                    "oracle_expected_mean_resource_gain": safe_mean(
                        [
                            row["oracle_expected_mastery_gain"]
                            for row in topic_resource_oracles
                        ]
                    ),
                    "model_use_allowed": False,
                }
                topic_oracle_rows.append(topic_oracle)
                all_topic_oracles.append(topic_oracle)

                selected_topic_flag = int(topic_id == selected_topic)
                observed_gain = (
                    float(event.score_gain)
                    if selected_topic_flag and pd.notna(event.score_gain)
                    else np.nan
                )
                #create a row for the topic candidate table, which contains only information 
                # available before the event, and is used for model training and evaluation
                topic_feature_rows.append(
                    {
                        "event_id": event_id,
                        "interaction_id": str(event.interaction_id),
                        "learner_id": learner_id,
                        "sequence_number": int(event.sequence_number),
                        "simulated_day": int(event.simulated_day),
                        "candidate_topic_id": topic_id,
                        "selected_topic_flag": selected_topic_flag,
                        "chronological_split": str(event.chronological_split),
                        "current_score": current_score,
                        "weakness_score": weakness,
                        "prior_interaction_count": int(overall["count"]),
                        "prior_completion_rate": float(overall["completion_rate"]),
                        "prior_mean_observed_gain": float(
                            overall["mean_observed_gain"]
                        ),
                        "topic_prior_interaction_count": int(topic_history["count"]),
                        "topic_prior_completion_rate": float(
                            topic_history["completion_rate"]
                        ),
                        "topic_prior_mean_observed_gain": float(
                            topic_history["mean_observed_gain"]
                        ),
                        "topic_seen_before": seen_before,
                        "days_since_topic_interaction": int(days_since_topic),
                        "cold_start_flag": int(len(history) < 3),
                        "prerequisite_count": len(prerequisite_ids),
                        "prerequisite_min_score": prerequisite_min_score,
                        "prerequisite_mean_score": prerequisite_mean_score,
                        "unmet_prerequisite_count": unmet_prerequisite_count,
                        "prerequisite_ready_60": int(
                            unmet_prerequisite_count == 0
                        ),
                        "downstream_mean_weakness": downstream_mean_weakness,
                        "downstream_max_weakness": downstream_max_weakness,
                        "topic_curriculum_order": int(topic["curriculum_order"]),
                        "topic_relative_difficulty": int(
                            topic["relative_difficulty"]
                        ),
                        "prerequisite_in_degree": int(
                            topic["prerequisite_in_degree"]
                        ),
                        "dependent_out_degree": int(topic["dependent_out_degree"]),
                        "downstream_topic_count": int(
                            topic["downstream_topic_count"]
                        ),
                        "reversed_pagerank": float(topic["reversed_pagerank"]),
                        "resource_count": int(len(candidate_resources)),
                        "resource_format_count": int(
                            candidate_resources["format_group"].nunique()
                        ),
                        "resource_min_difficulty_gap": float(min(difficulty_gaps)),
                        "resource_mean_difficulty_gap": safe_mean(difficulty_gaps),
                        "resource_max_tfidf_similarity": float(
                            max(similarities_for_topic)
                        ),
                        "resource_mean_tfidf_similarity": safe_mean(
                            similarities_for_topic
                        ),
                        "resource_mean_duration_minutes": float(
                            candidate_resources["duration_minutes"].mean()
                        ),
                        "catalogue_format_history_completion": safe_mean(
                            historical_format_completion
                        ),
                        "catalogue_format_history_rating": safe_mean(
                            historical_format_rating
                        ),
                        "observed_score_gain": observed_gain,
                    }
                )

            # Update state only after every candidate has been constructed
            if int(event.outcome_observed) == 1 and pd.notna(event.post_score):
                current_scores[selected_topic] = float(event.post_score)
            current_mastery[selected_topic] = float(event.latent_mastery_after)
            selected_resource_row = resources.loc[
                resources["resource_id"] == selected_resource
            ].iloc[0]
            history.append(
                {
                    "target_topic_id": selected_topic,
                    "format_group": str(selected_resource_row["format_group"]),
                    "completed": int(event.completed),
                    "score_gain": (
                        float(event.score_gain) if pd.notna(event.score_gain) else np.nan
                    ),
                    "usefulness_rating": (
                        float(event.usefulness_rating)
                        if pd.notna(event.usefulness_rating)
                        else np.nan
                    ),
                    "simulated_day": int(event.simulated_day),
                }
            )
            last_topic_day[selected_topic] = int(event.simulated_day)
    # store the reconstructed candidate features and oracle values for each event
    topic_features = pd.DataFrame(topic_feature_rows)
    resource_features = pd.DataFrame(resource_feature_rows)
    topic_oracle = pd.DataFrame(topic_oracle_rows)
    resource_oracle = pd.DataFrame(resource_oracle_rows)

    topic_oracle.loc[:, "oracle_topic_rank"] = (
        topic_oracle.groupby("event_id")[
            "oracle_expected_best_resource_gain"
        ].rank(method="first", ascending=False)
    )
    # the resource oracle rank is computed within each event and topic, based on the expected mastery gain
    resource_oracle.loc[:, "oracle_resource_rank_within_topic"] = (
        resource_oracle.groupby(["event_id", "candidate_topic_id"])[
            "oracle_expected_mastery_gain"
        ].rank(method="first", ascending=False)
    )

    original_comparison = original_resource_oracle.merge(
        resource_oracle,
        left_on=["event_id", "target_topic_id", "candidate_resource_id"],
        right_on=["event_id", "candidate_topic_id", "candidate_resource_id"],
        how="left",
        suffixes=("_original", "_rebuilt"),
        validate="one_to_one",
    )
    oracle_difference = (
        original_comparison["oracle_expected_mastery_gain_original"]
        - original_comparison["oracle_expected_mastery_gain_rebuilt"]
    ).abs()
    # Given the sensitive nature of the oracle values, we enforce a strict validation to ensure that
    #  the rebuilt values match the original ones within a very small tolerance. Any significant 
    # discrepancies would indicate a potential issue in the reconstruction process
    public_prohibited_tokens = (
        "oracle",
        "latent",
        "affinity",
        "quality",
        "appeal",
        "mastery",
    )
    public_columns = list(topic_features.columns) + list(resource_features.columns)
    prohibited_public_columns = sorted(
        {
            column
            for column in public_columns
            if any(token in column.lower() for token in public_prohibited_tokens)
        }
    )
    validation = {
        "events": int(events["event_id"].nunique()),
        "topic_candidate_rows": int(len(topic_features)),
        "resource_candidate_rows": int(len(resource_features)),
        "ten_topics_per_event": bool(
            topic_features.groupby("event_id").size().eq(len(topic_ids)).all()
        ),
        "forty_resources_per_event": bool(
            resource_features.groupby("event_id").size().eq(len(resources)).all()
        ),
        "four_resources_per_event_topic": bool(
            resource_features.groupby(["event_id", "candidate_topic_id"])
            .size()
            .eq(4)
            .all()
        ),
        "one_selected_topic_per_event": bool(
            topic_features.groupby("event_id")["selected_topic_flag"].sum().eq(1).all()
        ),
        "one_selected_resource_per_event": bool(
            resource_features.groupby("event_id")["selected_resource_flag"]
            .sum()
            .eq(1)
            .all()
        ),
    # the maximum difference between the selected score state and the original pre-event s
    # core is computed to ensure that the reconstruction process accurately reflects the learner's 
    # state before each event. Any significant discrepancies would indicate a potential issue in the 
    # reconstruction logic
        "maximum_selected_score_state_difference": float(
            max(selected_score_differences)
        ),
        "maximum_selected_mastery_state_difference": float(
            max(mastery_state_differences)
        ),
        "maximum_original_oracle_rebuild_difference": float(
            oracle_difference.max()
        ),
        "all_original_selected_topic_oracles_matched": bool(
            oracle_difference.notna().all() and oracle_difference.max() < 5e-8
        ),
        "prohibited_generator_truth_columns_in_public_tables": prohibited_public_columns,
        "graph_features_vary_across_topics": bool(
            topic_features.groupby("event_id")["reversed_pagerank"]
            .nunique()
            .gt(1)
            .all()
        ),
        "claim_boundary": (
            "Oracle tables validate ranking only and must never be used as model inputs. "
            "All conclusions remain limited to the controlled simulator."
        ),
    }
    # enforce strict validation rules to ensure the integrity of the candidate tables and oracle values
    required_true = [
        "ten_topics_per_event",
        "forty_resources_per_event",
        "four_resources_per_event_topic",
        "one_selected_topic_per_event",
        "one_selected_resource_per_event",
        "all_original_selected_topic_oracles_matched",
        "graph_features_vary_across_topics",
    ]
    failures = [name for name in required_true if not validation[name]]
    if prohibited_public_columns:
        failures.append("prohibited_generator_truth_columns_in_public_tables")
    if validation["maximum_selected_score_state_difference"] > 1e-8:
        failures.append("selected_score_state_reconstruction")
    if validation["maximum_selected_mastery_state_difference"] > 5e-8:
        failures.append("selected_mastery_state_reconstruction")
    if failures:
        raise AssertionError(f"Two-stage candidate validation failed: {failures}")

    return topic_features, resource_features, topic_oracle, resource_oracle, validation


def write_data_dictionary(
    topic_features: pd.DataFrame,
    resource_features: pd.DataFrame,
    topic_oracle: pd.DataFrame,
    resource_oracle: pd.DataFrame,
    destination: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for table_name, frame, table_role in [
        ("topic_candidates", topic_features, "public_pre_event_features"),
        ("resource_candidates", resource_features, "public_pre_event_features"),
        ("topic_candidate_oracle", topic_oracle, "evaluation_only_generator_truth"),
        (
            "resource_candidate_oracle",
            resource_oracle,
            "evaluation_only_generator_truth",
        ),
    ]:
        for column in frame.columns:
            if column == "observed_score_gain":
                role = "training_target_selected_events_only"
            elif column.endswith("_id") or column in {
                "event_id",
                "interaction_id",
                "learner_id",
            }:
                role = "identifier_or_grouping"
            elif "oracle" in column or table_role.startswith("evaluation"):
                role = "evaluation_only_never_model_input"
            elif column in {
                "selected_topic_flag",
                "selected_resource_flag",
                "chronological_split",
            }:
                role = "split_or_audit"
            else:
                role = "candidate_predictor"
            rows.append(
                {
                    "table": table_name,
                    "field": column,
                    "dtype": str(frame[column].dtype),
                    "table_role": table_role,
                    "field_role": role,
                }
            )
    pd.DataFrame(rows).to_csv(destination, index=False)

# hree the candidate tables and oracle tables are built, validated, and saved to the 
# specified directories. The function returns a validation dictionary summarizing the 
# results of the validation checks.
def save_candidate_tables(
    project_root: Path | str = DEFAULT_PROJECT_ROOT,
    dataset_root: Path | str | None = None,
) -> dict[str, Any]:
    # Build, validate and save the two candidate tables and oracle tables

    project_root = Path(project_root).resolve()
    public_dir = project_root / "two_stage_data"
    truth_dir = project_root / "two_stage_internal_truth"
    validation_dir = project_root / "two_stage_validation"
    public_dir.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)
    (
        topic_features,
        resource_features,
        topic_oracle,
        resource_oracle,
        validation,
    ) = build_candidate_tables(project_root, dataset_root)
    topic_features.to_csv(
        public_dir / "topic_candidates.csv", index=False, float_format="%.10f"
    )
    resource_features.to_csv(
        public_dir / "resource_candidates.csv", index=False, float_format="%.10f"
    )
    topic_oracle.to_csv(
        truth_dir / "topic_candidate_oracle.csv",
        index=False,
        float_format="%.10f",
    )
    resource_oracle.to_csv(
        truth_dir / "resource_candidate_oracle.csv",
        index=False,
        float_format="%.10f",
    )
    write_data_dictionary(
        topic_features,
        resource_features,
        topic_oracle,
        resource_oracle,
        public_dir / "two_stage_data_dictionary.csv",
    )
    (validation_dir / "candidate_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return validation

# keeps the command-line interface simple, allowing users to specify the project root and
#  dataset root directories. It returns a namespace containing the parsed arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Folder containing the source files and config.json",
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
    validation = save_candidate_tables(args.project_root, args.dataset_root)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
