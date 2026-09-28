#!/usr/bin/env python3
""" This is a controlled dataset generator for the CAPE project.  It is not a fitted model
and does not claim to reproduce the behaviour of real learners.  It is intended to
support development of the CAPE recommendation models and to provide a reproducible
benchmark for model evaluation.  The generator is seeded so that the same configuration
produces the same dataset each time.  The generator is not intended to be a realistic
simulator of learner behaviour and does not attempt to reproduce the full complexity
of real learners  

The generator keeps observable records separate from simulator-only truth,
preserves the order of learner interactions and produces validation evidence
alongside the data 




References
-------------------------
Salton and Buckley (1988) informed the use of TF-IDF for term weighting
https://hdl.handle.net/1813/6721


The scikit-learn documentation was used to confirm the behaviour of
TfidfVectorizer and cosine_similarity
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html


Page et al (1999) informed the iterative PageRank calculation
https://www.cis.upenn.edu/~mkearns/teaching/NetworkedLife/pagerank.pdf


Corbett and Anderson (1995) motivated the separation of unobserved knowledge
state from observed assessment results  The simulator is not a fitted BKT model
https://link.springer.com/article/10.1007/BF01099821


Stavinova et al (2022) informed the disclosure of simulator assumptions and the
separation of simulator truth from model inputs
https://arxiv.org/abs/2206.11338


Generative AI assisted with documentation review and comment editing.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "seed_data"


# ---------------------------------------------------------------------------
# Small numerical and file helpers
# ---------------------------------------------------------------------------


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    # Logistic transformation p = 1 / (1 + exp(-z))
    # Clipping avoids numerical overflow for very large positive or negative logits
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    # Temperature controls how strongly the largest value dominates the draw
    # Subtracting the maximum is the usual stability step before exponentiation
    scaled = np.asarray(values, dtype=float) / max(float(temperature), 1e-6)
    scaled -= scaled.max()
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()


def safe_mean(values: list[float], default: float) -> float:
    # Empty histories are expected for new learners so return an explicit fallback
    clean = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(clean)) if clean else float(default)


def file_sha256(path: Path) -> str:
    # Read in blocks so a large result file is not loaded into memory at once
    # Python reference consulted https://docs.python.org/3/library/hashlib.html
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Input loading and curriculum graph construction
# ---------------------------------------------------------------------------

#this function loads the configuration and seed tables for the dataset generator
def load_inputs(config_path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Seed tables are human-prepared inputs while learner records are generated later
    config = json.loads(config_path.read_text(encoding="utf-8"))
    topics = pd.read_csv(SEED_DIR / "topics.csv")
    prerequisites = pd.read_csv(SEED_DIR / "prerequisites.csv")
    resources = pd.read_csv(SEED_DIR / "resources.csv")
    sources = pd.read_csv(SEED_DIR / "method_sources.csv")
    return config, topics, prerequisites, resources, sources


# this function is used to check for cycles in the prerequisite graph and to produce a topological order of topics
def topological_order(topic_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    # Queue-based topological sorting also gives a direct cycle check
    # Definition checked against https://docs.python.org/3/library/graphlib.html
    adjacency: dict[str, list[str]] = {topic_id: [] for topic_id in topic_ids}
    indegree = {topic_id: 0 for topic_id in topic_ids}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    queue = deque(sorted([node for node, degree in indegree.items() if degree == 0]))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(topic_ids):
        raise ValueError("Prerequisite graph contains a cycle.")
    return order

# here we calculate graph features for each topic, including in-degree, out-degree, total degree centrality, 
# downstream topic count, and reversed PageRank
def graph_features(topics: pd.DataFrame, prerequisites: pd.DataFrame) -> pd.DataFrame:
    # A complete order exists only when the proposed prerequisite graph is acyclic
    topic_ids = topics["topic_id"].tolist()
    edges = list(prerequisites[["prerequisite_topic_id", "dependent_topic_id"]].itertuples(index=False, name=None))
    topological_order(topic_ids, edges)
    adjacency: dict[str, set[str]] = {topic_id: set() for topic_id in topic_ids}
    for source, target in edges:
        adjacency[source].add(target)

    def descendants(node: str) -> set[str]:
        found: set[str] = set()
        stack = list(adjacency[node])
        while stack:
            current = stack.pop()
            if current not in found:
                found.add(current)
                stack.extend(adjacency[current])
        return found

    downstream = {node: len(descendants(node)) for node in topic_ids}
    in_degree = {node: 0 for node in topic_ids}
    out_degree = {node: len(adjacency[node]) for node in topic_ids}
    for source, target in edges:
        in_degree[target] += 1

    # Reverse dependent-to-prerequisite edges so foundational topics receive rank
    # r_i = (1 - alpha) / N + alpha multiplied by incoming rank shares
    # Method reference Page et al (1999) http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
    reversed_adj: dict[str, set[str]] = {node: set() for node in topic_ids}
    for prerequisite, dependent in edges:
        reversed_adj[dependent].add(prerequisite)
    alpha = 0.85
    n_nodes = len(topic_ids)
    rank = {node: 1.0 / n_nodes for node in topic_ids}
    # Stop early once successive rank vectors differ by less than the tolerance
    for _ in range(200):
        next_rank = {node: (1.0 - alpha) / n_nodes for node in topic_ids}
        dangling = sum(rank[node] for node in topic_ids if not reversed_adj[node])
        for node in topic_ids:
            next_rank[node] += alpha * dangling / n_nodes
        for source in topic_ids:
            if reversed_adj[source]:
                share = alpha * rank[source] / len(reversed_adj[source])
                for target in reversed_adj[source]:
                    next_rank[target] += share
        delta = sum(abs(next_rank[node] - rank[node]) for node in topic_ids)
        rank = next_rank
        if delta < 1e-12:
            break

    rows = []
    for topic_id in topic_ids:
        rows.append(
            {
                "topic_id": topic_id,
                "prerequisite_in_degree": in_degree[topic_id],
                "dependent_out_degree": out_degree[topic_id],
                "total_degree_centrality": (in_degree[topic_id] + out_degree[topic_id]) / max(n_nodes - 1, 1),
                "downstream_topic_count": downstream[topic_id],
                "reversed_pagerank": rank[topic_id],
            }
        )
    return pd.DataFrame(rows)

# this function builds a TF-IDF mapping between topics and resources, returning a DataFrame of similarities and a lookup dictionary
def build_tfidf_mapping(topics: pd.DataFrame, resources: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    # Topic and resource fields are joined into short documents for a lexical audit
    # This does not discover curriculum meaning and does not replace the researcher mapping
    topic_docs = (
        topics["topic_name"].fillna("")
        + " "
        + topics["syllabus_objective_text"].fillna("")
    ).tolist()
    resource_docs = (
        resources["title"].fillna("")
        + " "
        + resources["description"].fillna("")
        + " "
        + resources["learning_objective"].fillna("")
        + " "
        + resources["language_alignment"].fillna("")
    ).tolist()
    # TF-IDF follows Salton and Buckley (1988) while the parameters follow scikit-learn
    # Bigrams retain short phrases and sublinear_tf reduces the influence of repetition
    # https://hdl.handle.net/1813/6721
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(topic_docs + resource_docs)

    # Cosine similarity compares vector direction rather than document length
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    similarities = cosine_similarity(matrix[len(topic_docs) :], matrix[: len(topic_docs)])
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], float] = {}
    for resource_index, resource in resources.reset_index(drop=True).iterrows():
        for topic_index, topic in topics.reset_index(drop=True).iterrows():
            score = float(similarities[resource_index, topic_index])
            lookup[(resource["resource_id"], topic["topic_id"])] = score
            rows.append(
                {
                    "resource_id": resource["resource_id"],
                    "topic_id": topic["topic_id"],
                    "tfidf_cosine_similarity": score,
                    "is_primary_researcher_mapping": int(resource["primary_topic_id"] == topic["topic_id"]),
                    "mapping_validation_status": "pending_educator_review",
                }
            )
    return pd.DataFrame(rows), lookup


# ---------------------------------------------------------------------------
# Learner profiles and assessment helpers
# ---------------------------------------------------------------------------

# THis function assigns learners to profiles based on the specified proportions, ensuring that the total 
# number of learners matches n_learners
def make_profile_assignments(rng: np.random.Generator, n_learners: int, proportions: dict[str, float]) -> list[str]:
    # Largest fractional remainders receive the spare places so totals remain exact
    names = list(proportions)
    raw_counts = np.array([proportions[name] * n_learners for name in names])
    counts = np.floor(raw_counts).astype(int)
    remainder = n_learners - int(counts.sum())
    for index in np.argsort(-(raw_counts - counts))[:remainder]:
        counts[index] += 1
    profiles = [name for name, count in zip(names, counts) for _ in range(int(count))]
    rng.shuffle(profiles)
    return profiles

# This function samples learner traits based on the specified profile and formats, returning a dictionary of traits
def sample_learner_traits(rng: np.random.Generator, profile: str, formats: list[str]) -> dict[str, Any]:
    # These distributions are disclosed project assumptions rather than CAPE estimates
    # Profile labels exist for stress testing and are excluded from model predictors
    if profile == "high_mastery_low_confidence":
        ability = 0.62 + 0.36 * rng.beta(7.0, 2.0)
        engagement = rng.beta(4.0, 2.5)
        responsiveness = np.clip(rng.normal(1.05, 0.12), 0.65, 1.40)
        confidence_bias = rng.normal(-1.80, 0.20)
    elif profile == "low_mastery_high_confidence":
        ability = 0.08 + 0.48 * rng.beta(2.2, 5.5)
        engagement = rng.beta(3.5, 3.0)
        responsiveness = np.clip(rng.normal(0.95, 0.15), 0.55, 1.35)
        confidence_bias = rng.normal(1.70, 0.20)
    elif profile == "capable_low_engagement":
        ability = 0.50 + 0.42 * rng.beta(5.0, 2.2)
        engagement = rng.beta(1.4, 5.0)
        responsiveness = np.clip(rng.normal(1.00, 0.14), 0.60, 1.35)
        confidence_bias = rng.normal(0.00, 0.35)
    elif profile == "persistent_support_need":
        ability = 0.08 + 0.48 * rng.beta(2.0, 5.2)
        engagement = rng.beta(3.0, 3.2)
        responsiveness = np.clip(rng.normal(0.72, 0.09), 0.45, 0.95)
        confidence_bias = rng.normal(-0.20, 0.40)
    else:
        ability = rng.beta(5.0, 4.0)
        engagement = rng.beta(4.0, 3.0)
        responsiveness = np.clip(rng.normal(1.00, 0.15), 0.55, 1.45)
        confidence_bias = rng.normal(0.00, 0.45)
    # Dirichlet draws create relative preferences that add to one across formats
    # They are noisy behavioural tendencies and must not be read as learning styles
    format_affinity_values = rng.dirichlet(np.full(len(formats), 2.0))
    return {
        "general_ability": float(np.clip(ability, 0.04, 0.97)),
        "engagement_propensity": float(np.clip(engagement, 0.03, 0.98)),
        "learning_responsiveness": float(responsiveness),
        "confidence_bias": float(confidence_bias),
        "individual_process_noise": float(np.clip(rng.lognormal(mean=-4.4, sigma=0.25), 0.007, 0.025)),
        "format_affinity": dict(zip(formats, format_affinity_values.astype(float))),
    }


# assessment simulation is a project-specific diagnostic model that converts latent mastery and 
# item difficulty into observable scores
def simulate_assessment(
    rng: np.random.Generator,
    mastery: float,
    topic_relative_difficulty: int,
    diagnostic_config: dict[str, Any],
    item_thresholds: np.ndarray | None = None,
) -> tuple[int, int, float, np.ndarray]:
    # A logistic response curve turns latent mastery and item difficulty into probability
    # This is a project-specific diagnostic model and not a fitted IRT or BKT model
    # Corbett and Anderson (1995) only motivates the hidden changing knowledge state
    offsets = np.asarray(diagnostic_config["item_difficulty_offsets"], dtype=float)
    n_items = int(diagnostic_config["items_per_topic"])
    if len(offsets) != n_items:
        raise ValueError("The diagnostic item difficulty offset count must equal items_per_topic.")
    centre = 0.50 + 0.07 * (int(topic_relative_difficulty) - 2)
    item_difficulties = np.clip(centre + offsets, 0.08, 0.92)
    probabilities = sigmoid(float(diagnostic_config["discrimination"]) * (mastery - item_difficulties))
    # Reusing the same base thresholds at post-test keeps item identity stable
    if item_thresholds is None:
        item_thresholds = rng.uniform(0.0, 1.0, n_items)
    occasion_thresholds = np.clip(
        item_thresholds
        + rng.normal(0.0, float(diagnostic_config["occasion_threshold_noise_sd"]), n_items),
        0.0,
        1.0,
    )
    correct = int((occasion_thresholds < probabilities).sum())
    score = 100.0 * correct / n_items
    return correct, n_items, float(score), np.asarray(item_thresholds, dtype=float)


def confidence_from_mastery(rng: np.random.Generator, mastery: float, bias: float) -> float:
    # Confidence is deliberately imperfect because self-report need not equal performance
    value = 1.0 + 4.0 * mastery + bias + rng.normal(0.0, 0.30)
    return float(np.round(np.clip(value, 1.0, 5.0), 1))


def desired_resource_difficulty(pre_score: float) -> int:
    # The 45 and 75 cut points are declared project rules rather than empirical findings
    if pre_score < 45.0:
        return 1
    if pre_score < 75.0:
        return 2
    return 3


# this function calculates prior history features for a learner, given their interaction history, topic ID, and format group
def prior_history_features(history: list[dict[str, Any]], topic_id: str, format_group: str) -> dict[str, Any]:
    # Only already completed events enter these summaries to avoid future-data leakage
    topic_history = [row for row in history if row["target_topic_id"] == topic_id]
    format_history = [row for row in history if row["format_group"] == format_group]
    observed_gains = [row["score_gain"] for row in history if pd.notna(row["score_gain"])]
    topic_gains = [row["score_gain"] for row in topic_history if pd.notna(row["score_gain"])]
    format_ratings = [row["usefulness_rating"] for row in format_history if pd.notna(row["usefulness_rating"])]
    return {
        "prior_interaction_count": len(history),
        "prior_completion_rate": safe_mean([row["completed"] for row in history], 0.0),
        "prior_mean_observed_gain": safe_mean(observed_gains, 0.0),
        "topic_prior_interaction_count": len(topic_history),
        "topic_prior_mean_observed_gain": safe_mean(topic_gains, 0.0),
        "format_prior_interaction_count": len(format_history),
        "format_prior_completion_rate": safe_mean([row["completed"] for row in format_history], 0.0),
        "format_prior_mean_rating": safe_mean(format_ratings, 0.0),
        "cold_start_flag": int(len(history) < 3),
    }


# this function generates the dataset for the CAPE project, simulating learner interactions with 
# topics and resources based on the provided configuration and seed data
def generate_dataset(config: dict[str, Any], topics: pd.DataFrame, prerequisites: pd.DataFrame, resources: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # One seeded Generator controls every random draw in the simulation
    # NumPy reference https://numpy.org/doc/stable/reference/random/generator.html
    rng = np.random.default_rng(int(config["random_seed"]))
    topic_ids = topics["topic_id"].tolist()
    topic_index = {topic_id: index for index, topic_id in enumerate(topic_ids)}
    topic_difficulty = dict(zip(topics["topic_id"], topics["relative_difficulty"].astype(int)))
    topic_order = dict(zip(topics["topic_id"], topics["curriculum_order"].astype(int)))
    prerequisite_map: dict[str, list[str]] = defaultdict(list)
    dependent_map: dict[str, list[str]] = defaultdict(list)
    for prerequisite, dependent in prerequisites[["prerequisite_topic_id", "dependent_topic_id"]].itertuples(index=False, name=None):
        prerequisite_map[dependent].append(prerequisite)
        dependent_map[prerequisite].append(dependent)

    # Static curriculum and text features are calculated once before learner simulation
    graph = graph_features(topics, prerequisites)
    graph_lookup = graph.set_index("topic_id").to_dict("index")
    similarities, similarity_lookup = build_tfidf_mapping(topics, resources)
    resources_by_topic = {
        topic_id: group.reset_index(drop=True)
        for topic_id, group in resources.groupby("primary_topic_id", sort=False)
    }

    formats = list(config["formats"])
    profiles = make_profile_assignments(rng, int(config["n_learners"]), config["learner_profiles"])
    learner_rows: list[dict[str, Any]] = []
    learner_truth_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    baseline_truth_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    interaction_truth_rows: list[dict[str, Any]] = []
    potential_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    resource_truth_rows: list[dict[str, Any]] = []

    # Hidden resource quality and appeal are used only by the simulator
    # They are saved separately and never supplied to the recommendation models
    resource_truth: dict[str, dict[str, float]] = {}
    for resource in resources.itertuples(index=False):
        quality = float(0.48 + 0.48 * rng.beta(5.0, 2.2))
        appeal = float(0.15 + 0.75 * rng.beta(3.0, 3.0))
        resource_truth[resource.resource_id] = {"latent_quality": quality, "latent_appeal": appeal}
        resource_truth_rows.append(
            {
                "resource_id": resource.resource_id,
                "latent_quality": quality,
                "latent_appeal": appeal,
                "model_use_allowed": False,
            }
        )

    # Learners are generated independently but share the same curriculum catalogue
    global_interaction_number = 0
    for learner_number in range(1, int(config["n_learners"]) + 1):
        learner_id = f"L{learner_number:04d}"
        profile = profiles[learner_number - 1]
        traits = sample_learner_traits(rng, profile, formats)
        # The Poisson draw varies history length while configured bounds prevent extremes
        raw_interactions = int(config["interaction_count"]["minimum"]) + int(
            rng.poisson(float(config["interaction_count"]["poisson_lambda"]))
        )
        interaction_budget = int(
            np.clip(
                raw_interactions,
                int(config["interaction_count"]["minimum"]),
                int(config["interaction_count"]["maximum"]),
            )
        )
        learner_rows.append(
            {
                "learner_id": learner_id,
                "simulation_cohort": config["scenario"],
                "audit_profile_group": profile,
                "planned_interaction_count": interaction_budget,
                "scale_100_member": int(learner_number <= 100),
                "scale_250_member": int(learner_number <= 250),
                "scale_500_member": 1,
                "audit_profile_model_use_allowed": False,
            }
        )
        truth_row: dict[str, Any] = {
            "learner_id": learner_id,
            "simulation_profile": profile,
            "general_ability": traits["general_ability"],
            "engagement_propensity": traits["engagement_propensity"],
            "learning_responsiveness": traits["learning_responsiveness"],
            "confidence_bias": traits["confidence_bias"],
            "individual_process_noise": traits["individual_process_noise"],
            "model_use_allowed": False,
        }
        for format_group, affinity in traits["format_affinity"].items():
            truth_row[f"affinity_{format_group}"] = affinity
        learner_truth_rows.append(truth_row)

        # Latent mastery is simulator truth while scores and confidence are observable
        mastery: dict[str, float] = {}
        observed_score: dict[str, float] = {}
        confidence: dict[str, float] = {}
        assessment_thresholds: dict[str, np.ndarray] = {}
        for topic in topics.itertuples(index=False):
            topic_offset = 0.055 * (int(topic.relative_difficulty) - 2)
            profile_offset = -0.07 if profile == "persistent_support_need" else 0.0
            initial_mastery = float(
                np.clip(
                    0.78 * traits["general_ability"]
                    + 0.22 * rng.beta(3.0, 3.0)
                    - topic_offset
                    + profile_offset
                    + rng.normal(0.0, 0.085),
                    0.04,
                    0.96,
                )
            )
            mastery[topic.topic_id] = initial_mastery
            correct, item_count, score, stable_thresholds = simulate_assessment(
                rng,
                initial_mastery,
                int(topic.relative_difficulty),
                config["diagnostic"],
            )
            assessment_thresholds[topic.topic_id] = stable_thresholds
            confidence_value = confidence_from_mastery(rng, initial_mastery, traits["confidence_bias"])
            observed_score[topic.topic_id] = score
            confidence[topic.topic_id] = confidence_value
            baseline_rows.append(
                {
                    "learner_id": learner_id,
                    "topic_id": topic.topic_id,
                    "diagnostic_correct_items": correct,
                    "diagnostic_item_count": item_count,
                    "diagnostic_score": score,
                    "confidence_score": confidence_value,
                }
            )
            baseline_truth_rows.append(
                {
                    "learner_id": learner_id,
                    "topic_id": topic.topic_id,
                    "initial_latent_mastery": initial_mastery,
                    "model_use_allowed": False,
                }
            )

        # History starts empty and is updated only after each event has been completed
        history: list[dict[str, Any]] = []
        topic_event_counts = defaultdict(int)
        simulated_day = int(rng.integers(0, 8))
        for sequence_number in range(1, interaction_budget + 1):
            simulated_day += int(np.clip(rng.poisson(3.0) + 1, 1, 10))
            # Weakness is the complement of the latest observed score
            weakness = {topic_id: 1.0 - observed_score[topic_id] / 100.0 for topic_id in topic_ids}
            priority_values: list[float] = []
            for topic_id in topic_ids:
                downstream_nodes: list[str] = []
                queue = list(dependent_map[topic_id])
                while queue:
                    node = queue.pop()
                    if node not in downstream_nodes:
                        downstream_nodes.append(node)
                        queue.extend(dependent_map[node])
                downstream_need = safe_mean([weakness[node] for node in downstream_nodes], 0.0)
                prerequisite_support = weakness[topic_id] * downstream_need
                graph_value = graph_lookup[topic_id]["downstream_topic_count"] / max(len(topic_ids) - 1, 1)
                selection_config = config["target_topic_selection"]
                # Topic priority combines current weakness, graph reach and repeat control
                priority = (
                    float(selection_config["weakness_weight"]) * weakness[topic_id]
                    + float(selection_config["downstream_weight"]) * graph_value
                    + float(selection_config["prerequisite_support_weight"]) * prerequisite_support
                    - float(selection_config["repeat_topic_penalty"]) * min(topic_event_counts[topic_id], 3)
                    + rng.normal(0.0, 0.025)
                )
                priority_values.append(priority)
            eligible_count = min(int(config["target_topic_selection"]["eligible_topic_count"]), len(topic_ids))
            eligible_indices = np.argsort(priority_values)[-eligible_count:]
            eligible_priorities = np.asarray([priority_values[index] for index in eligible_indices])
            # Draw from the strongest eligible topics rather than always taking one maximum
            topic_probabilities = softmax(
                eligible_priorities,
                float(config["target_topic_selection"]["softmax_temperature"]),
            )
            selected_eligible_index = int(rng.choice(len(eligible_indices), p=topic_probabilities))
            target_topic_id = topic_ids[int(eligible_indices[selected_eligible_index])]
            topic_event_counts[target_topic_id] += 1
            pre_score = float(observed_score[target_topic_id])
            confidence_before = float(confidence[target_topic_id])
            desired_difficulty = desired_resource_difficulty(pre_score)
            prerequisite_ids = prerequisite_map[target_topic_id]
            prerequisite_observed_scores = [observed_score[node] for node in prerequisite_ids]
            prerequisite_latent_values = [mastery[node] for node in prerequisite_ids]
            prerequisite_mean_score = safe_mean(prerequisite_observed_scores, 100.0)
            prerequisite_min_score = min(prerequisite_observed_scores) if prerequisite_observed_scores else 100.0
            prerequisite_readiness = safe_mean(prerequisite_latent_values, 1.0)
            unmet_prerequisite_count = sum(score < 60.0 for score in prerequisite_observed_scores)

            # Four resources belong to each topic in the prepared catalogue
            candidate_resources = resources_by_topic[target_topic_id]
            candidate_logits: list[float] = []
            candidate_context: list[dict[str, Any]] = []
            for candidate in candidate_resources.itertuples(index=False):
                difficulty_match = 1.0 - abs(int(candidate.difficulty) - desired_difficulty) / 2.0
                format_match = float(traits["format_affinity"][candidate.format_group])
                semantic_similarity = float(similarity_lookup[(candidate.resource_id, target_topic_id)])
                hidden = resource_truth[candidate.resource_id]
                selection_config = config["resource_selection"]
                # Selection is intentionally biased but still probabilistic
                # Hidden appeal affects generated behaviour but is excluded from modelling
                logit = (
                    float(selection_config["semantic_similarity_weight"]) * semantic_similarity
                    + float(selection_config["difficulty_match_weight"]) * difficulty_match
                    + float(selection_config["format_affinity_weight"]) * format_match
                    + float(selection_config["latent_appeal_weight"]) * hidden["latent_appeal"]
                )
                candidate_logits.append(logit)
                candidate_context.append(
                    {
                        "resource": candidate,
                        "difficulty_match": difficulty_match,
                        "format_match": format_match,
                        "semantic_similarity": semantic_similarity,
                        **hidden,
                    }
                )
            selection_probabilities = softmax(
                np.asarray(candidate_logits),
                float(config["resource_selection"]["softmax_temperature"]),
            )
            # Random exploration prevents the simulator from selecting only obvious matches
            exploration = bool(rng.random() < float(config["resource_selection"]["exploration_probability"]))
            if exploration:
                selected_candidate_index = int(rng.integers(0, len(candidate_context)))
                effective_selection_probability = 1.0 / len(candidate_context)
                selection_mode = "random_exploration"
            else:
                selected_candidate_index = int(rng.choice(len(candidate_context), p=selection_probabilities))
                effective_selection_probability = float(selection_probabilities[selected_candidate_index])
                selection_mode = "probabilistic_fit"

            # Evaluate all four possible resources before drawing the observed interaction
            # These counterfactual values are oracle evidence and never model inputs
            event_id = f"E{global_interaction_number + 1:06d}"
            oracle_values: list[float] = []
            for candidate_index, context in enumerate(candidate_context):
                candidate = context["resource"]
                duration_scaled = (float(candidate.duration_minutes) - 10.0) / 70.0
                overchallenge = max(0.0, int(candidate.difficulty) - desired_difficulty) / 2.0
                completion_config = config["completion"]
                # Completion probability uses a disclosed logistic model
                completion_logit = (
                    float(completion_config["intercept"])
                    + float(completion_config["engagement_weight"]) * traits["engagement_propensity"]
                    + float(completion_config["format_match_weight"]) * context["format_match"]
                    + float(completion_config["difficulty_match_weight"]) * context["difficulty_match"]
                    + float(completion_config["latent_appeal_weight"]) * context["latent_appeal"]
                    - float(completion_config["duration_penalty"]) * duration_scaled
                    - float(completion_config["overchallenge_penalty"]) * overchallenge
                )
                completion_probability = float(sigmoid(completion_logit))
                learning_config = config["latent_learning"]
                complete_effect = (
                    float(learning_config["base"])
                    + float(learning_config["resource_quality_weight"]) * context["latent_quality"]
                    + float(learning_config["format_match_weight"]) * context["format_match"]
                    + float(learning_config["difficulty_match_weight"]) * context["difficulty_match"]
                    + float(learning_config["prerequisite_readiness_weight"]) * prerequisite_readiness
                )
                # Remaining room for improvement creates a simple ceiling effect
                full_mastery_gain = (
                    complete_effect
                    * traits["learning_responsiveness"]
                    * max(0.02, 1.0 - mastery[target_topic_id])
                )
                incomplete_multiplier = float(learning_config["incomplete_exposure_multiplier"])
                # Expected gain averages completed and incomplete exposure outcomes
                expected_mastery_gain = full_mastery_gain * (
                    completion_probability + (1.0 - completion_probability) * incomplete_multiplier
                )
                expected_mastery_gain = float(
                    np.clip(
                        expected_mastery_gain,
                        0.0,
                        float(learning_config["maximum_single_interaction_mastery_gain"]),
                    )
                )
                oracle_values.append(expected_mastery_gain)
                potential_rows.append(
                    {
                        "event_id": event_id,
                        "learner_id": learner_id,
                        "sequence_number": sequence_number,
                        "target_topic_id": target_topic_id,
                        "candidate_resource_id": candidate.resource_id,
                        "selected_resource_flag": int(candidate_index == selected_candidate_index),
                        "oracle_completion_probability": completion_probability,
                        "oracle_expected_mastery_gain": expected_mastery_gain,
                        "oracle_expected_gain_points": 100.0 * expected_mastery_gain,
                        "selection_probability_if_exploiting": float(selection_probabilities[candidate_index]),
                        "model_use_allowed": False,
                    }
                )

            # Only the selected resource becomes part of the public interaction history
            selected = candidate_context[selected_candidate_index]
            selected_resource = selected["resource"]
            selected_potential_index = len(potential_rows) - len(candidate_context) + selected_candidate_index
            selected_potential = potential_rows[selected_potential_index]
            completion_probability = float(selected_potential["oracle_completion_probability"])
            completed = int(rng.random() < completion_probability)
            duration_factor = float(
                np.clip(
                    rng.normal(0.42 + 0.52 * completed + 0.16 * traits["engagement_propensity"], 0.16),
                    0.05,
                    1.45,
                )
            )
            time_spent_minutes = float(np.round(float(selected_resource.duration_minutes) * duration_factor, 1))
            overchallenge = max(0.0, int(selected_resource.difficulty) - desired_difficulty) / 2.0
            attempts = int(1 + rng.poisson(0.35 + 1.10 * overchallenge + 0.25 * (1.0 - mastery[target_topic_id])))
            exposure_multiplier = 1.0 if completed else float(config["latent_learning"]["incomplete_exposure_multiplier"])
            expected_if_completed = float(selected_potential["oracle_expected_mastery_gain"]) / max(
                completion_probability
                + (1.0 - completion_probability) * float(config["latent_learning"]["incomplete_exposure_multiplier"]),
                1e-8,
            )
            process_noise = rng.normal(
                0.0,
                max(float(config["latent_learning"]["process_noise_sd"]), traits["individual_process_noise"]),
            )
            # Process noise prevents expected and realised mastery gain from being identical
            latent_mastery_gain = float(
                np.clip(
                    expected_if_completed * exposure_multiplier + process_noise,
                    0.0,
                    float(config["latent_learning"]["maximum_single_interaction_mastery_gain"]),
                )
            )
            mastery_before = float(mastery[target_topic_id])
            mastery_after = float(np.clip(mastery_before + latent_mastery_gain, 0.0, 0.995))
            mastery[target_topic_id] = mastery_after
            confidence_after = confidence_from_mastery(rng, mastery_after, traits["confidence_bias"])
            # A post-assessment is recorded only for a configured subset of completions
            outcome_observed = int(
                completed
                and rng.random() < float(config["diagnostic"]["post_assessment_probability_after_completion"])
            )
            if outcome_observed:
                post_correct, post_items, post_score, _ = simulate_assessment(
                    rng,
                    mastery_after,
                    topic_difficulty[target_topic_id],
                    config["diagnostic"],
                    item_thresholds=assessment_thresholds[target_topic_id],
                )
                # Report definition y_e = post_score_e - pre_score_e
                score_gain = float(post_score - pre_score)
                observed_score[target_topic_id] = post_score
            else:
                post_correct, post_items, post_score, score_gain = np.nan, np.nan, np.nan, np.nan
            if completed:
                confidence[target_topic_id] = confidence_after
            # Ratings are optional outcomes and remain missing when no rating is generated
            rating_observed = int(
                completed and rng.random() < float(config["rating"]["probability_after_completion"])
            )
            if rating_observed:
                normalised_gain = min(latent_mastery_gain / 0.09, 1.0)
                experienced_utility = (
                    0.32 * selected["latent_quality"]
                    + 0.20 * selected["format_match"]
                    + 0.18 * selected["difficulty_match"]
                    + 0.30 * normalised_gain
                )
                usefulness_rating = int(
                    np.clip(
                        np.rint(1.0 + 4.0 * experienced_utility + rng.normal(0.0, float(config["rating"]["noise_sd"]))),
                        1,
                        5,
                    )
                )
            else:
                usefulness_rating = np.nan

            global_interaction_number += 1
            interaction_id = f"I{global_interaction_number:06d}"
            # Public records contain observable fields only
            public_interaction = {
                "interaction_id": interaction_id,
                "learner_id": learner_id,
                "resource_id": selected_resource.resource_id,
                "sequence_number": sequence_number,
                "simulated_day": simulated_day,
                "target_topic_id": target_topic_id,
                "pre_score": pre_score,
                "confidence_before": confidence_before,
                "completed": completed,
                "time_spent_minutes": time_spent_minutes,
                "attempt_count": attempts,
                "usefulness_rating": usefulness_rating,
                "post_correct_items": post_correct,
                "post_item_count": post_items,
                "post_score": post_score,
                "score_gain": score_gain,
                "outcome_observed": outcome_observed,
            }
            interaction_rows.append(public_interaction)
            # Simulator-only values are written to a different table with model use disabled
            interaction_truth_rows.append(
                {
                    "interaction_id": interaction_id,
                    "event_id": event_id,
                    "selection_mode": selection_mode,
                    "effective_selection_probability": effective_selection_probability,
                    "latent_mastery_before": mastery_before,
                    "latent_mastery_after": mastery_after,
                    "latent_mastery_gain": latent_mastery_gain,
                    "latent_resource_quality": selected["latent_quality"],
                    "latent_resource_appeal": selected["latent_appeal"],
                    "hidden_format_affinity": selected["format_match"],
                    "oracle_completion_probability": completion_probability,
                    "oracle_expected_mastery_gain": selected_potential["oracle_expected_mastery_gain"],
                    "model_use_allowed": False,
                }
            )
            # Features are calculated before the current row is appended to history
            history_features = prior_history_features(history, target_topic_id, selected_resource.format_group)
            graph_values = graph_lookup[target_topic_id]
            model_row = {
                "interaction_id": interaction_id,
                "learner_id": learner_id,
                "resource_id": selected_resource.resource_id,
                "target_topic_id": target_topic_id,
                "sequence_number": sequence_number,
                "simulated_day": simulated_day,
                "pre_score": pre_score,
                "confidence_before": confidence_before,
                "weakness_score": 1.0 - pre_score / 100.0,
                **history_features,
                "prerequisite_count": len(prerequisite_ids),
                "prerequisite_min_score": prerequisite_min_score,
                "prerequisite_mean_score": prerequisite_mean_score,
                "unmet_prerequisite_count": unmet_prerequisite_count,
                "resource_format_group": selected_resource.format_group,
                "resource_difficulty": int(selected_resource.difficulty),
                "resource_duration_minutes": float(selected_resource.duration_minutes),
                "desired_resource_difficulty": desired_difficulty,
                "difficulty_gap": abs(int(selected_resource.difficulty) - desired_difficulty),
                "tfidf_cosine_similarity": selected["semantic_similarity"],
                "topic_curriculum_order": topic_order[target_topic_id],
                "topic_relative_difficulty": topic_difficulty[target_topic_id],
                "prerequisite_in_degree": graph_values["prerequisite_in_degree"],
                "dependent_out_degree": graph_values["dependent_out_degree"],
                "downstream_topic_count": graph_values["downstream_topic_count"],
                "reversed_pagerank": graph_values["reversed_pagerank"],
                "target_completed": completed,
                "target_score_gain": score_gain,
                "target_usefulness_rating": usefulness_rating,
            }
            model_rows.append(model_row)
            history.append(
                {
                    **public_interaction,
                    "target_topic_id": target_topic_id,
                    "format_group": selected_resource.format_group,
                }
            )

    interactions = pd.DataFrame(interaction_rows)
    model_ready_all = pd.DataFrame(model_rows)

    # Hold out the final events separately for every learner to preserve chronology
    # math.ceil keeps at least the configured fraction when histories are short
    chronological_split = pd.Series("train_history", index=interactions.index, dtype="object")
    holdout_fraction = float(config["chronological_holdout_fraction"])
    for learner_id, indices in interactions.groupby("learner_id").groups.items():
        ordered = list(indices)
        n_holdout = max(1, int(math.ceil(len(ordered) * holdout_fraction)))
        chronological_split.loc[ordered[-n_holdout:]] = "future_holdout"
    interactions = interactions.assign(chronological_split=chronological_split.to_numpy())
    split_lookup = interactions.set_index("interaction_id")["chronological_split"]

    # Copy the event split by identifier so interaction and model tables stay aligned
    model_ready_all = model_ready_all.assign(
        chronological_split=model_ready_all["interaction_id"].map(split_lookup)
    )

    # Supervised training requires an observed score-gain target
    model_ready = model_ready_all[model_ready_all["target_score_gain"].notna()].reset_index(drop=True)

    potential_outcomes = pd.DataFrame(potential_rows)
    potential_outcomes = potential_outcomes.assign(
        oracle_rank_within_event=(
            potential_outcomes.groupby("event_id")["oracle_expected_mastery_gain"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    )
    selected_truth = pd.DataFrame(interaction_truth_rows)
    learner_truth = pd.DataFrame(learner_truth_rows)
    learner_topic_truth = pd.DataFrame(baseline_truth_rows)
    learners = pd.DataFrame(learner_rows)
    baseline = pd.DataFrame(baseline_rows)
    resource_truth_df = pd.DataFrame(resource_truth_rows)
    topics_out = topics.merge(graph, on="topic_id", how="left", validate="one_to_one")
    resource_topics = resources[["resource_id", "primary_topic_id"]].rename(columns={"primary_topic_id": "topic_id"})
    resource_topics = resource_topics.merge(
        similarities,
        on=["resource_id", "topic_id"],
        how="left",
        validate="one_to_one",
    )
    resource_topics = resource_topics.assign(
        mapping_method="researcher_mapping_plus_tfidf_audit",
        educator_decision="pending",
    )

    # Return public and private tables together so validation can compare them
    return {
        "topics": topics_out,
        "prerequisites": prerequisites.copy(),
        "resources": resources.copy(),
        "resource_topics": resource_topics,
        "resource_topic_similarity": similarities,
        "learners": learners,
        "learner_topic_baseline": baseline,
        "interactions": interactions,
        "model_ready": model_ready,
        "model_ready_all_events": model_ready_all,
        "learner_truth": learner_truth,
        "learner_topic_truth": learner_topic_truth,
        "resource_truth": resource_truth_df,
        "selected_interaction_truth": selected_truth,
        "candidate_potential_outcomes": potential_outcomes,
    }


# ---------------------------------------------------------------------------
# Data dictionary descriptions
# ---------------------------------------------------------------------------

#here is a dictionary that provides human-readable descriptions for each field in the generated dataset. 
# This can be used for documentation, validation, and understanding the purpose of each column in the data tables
FIELD_DESCRIPTIONS: dict[str, str] = {
    "topic_id": "Stable pseudonymous topic identifier",
    "topic_name": "Human readable topic name",
    "syllabus_objective_text": "Researcher paraphrase of the selected syllabus objective",
    "curriculum_order": "Researcher-assigned display order within the selected scope",
    "relative_difficulty": "Researcher-assigned ordinal topic difficulty from 1 to 3, pending educator review",
    "scope_note": "Boundary and validation note for the topic record",
    "source_reference": "Source used to establish curriculum relevance",
    "prerequisite_in_degree": "Count of direct prerequisite topics pointing to this topic",
    "dependent_out_degree": "Count of topics that directly depend on this topic",
    "total_degree_centrality": "Total direct degree divided by the maximum possible degree",
    "downstream_topic_count": "Count of direct and indirect dependent topics",
    "reversed_pagerank": "PageRank on dependent-to-prerequisite edges; kept separate from downstream count",
    "prerequisite_topic_id": "Topic asserted to be needed first",
    "dependent_topic_id": "Topic that depends on the prerequisite topic",
    "edge_rationale": "Researcher reasoning for the provisional prerequisite edge",
    "edge_status": "Validation status of the prerequisite edge",
    "resource_id": "Stable resource identifier",
    "primary_topic_id": "Researcher-assigned primary topic for the resource.",
    "title": "Resource title shown by the provider",
    "url": "Public resource URL checked on the retrieval date",
    "provider": "Resource publisher or platform",
    "description": "Short researcher written description of the resource",
    "learning_objective": "Researcher-written statement of the intended use of the resource",
    "primary_format": "Specific presentation or activity type",
    "format_group": "Broad format group used by the simulator: text, practice, interactive or video",
    "difficulty": "Researcher-assigned resource difficulty from 1 to 3, pending educator review",
    "duration_minutes": "Researcher estimate of completion time, pending educator review",
    "access_status": "Access status checked on the retrieval date.",
    "licence_note": "Licence caution; free access does not imply an open licence",
    "language_alignment": "Whether the resource uses procedural C or is language-neutral",
    "retrieval_date": "Date on which the resource URL and access were checked",
    "verification_status": "Extent of URL and content verification",
    "metadata_status": "Human-validation status of assigned metadata",
    "is_primary_researcher_mapping": "1 when the row is the researcher's primary topic mapping.",
    "mapping_validation_status": "Human-validation status of an NLP mapping",
    "tfidf_cosine_similarity": "Cosine similarity between TF-IDf vectors for resource text and topic text.",
    "mapping_method": "Method used to propose and audit the topic mapping",
    "educator_decision": "Placeholder for educator confirmation or correction",
    "learner_id": "Pseudonymous synthetic learner identifier",
    "simulation_cohort": "Named generator scenario.",
    "audit_profile_group": "Synthetic stress-profile label for subgroup auditing only; excluded from model features.",
    "planned_interaction_count": "Number of interactions generated for the learner",
    "scale_100_member": "Deterministic membership flag for the 100-learner learning-curve subset",
    "scale_250_member": "Deterministic membership flag for the 250-learner learning-curve subset.",
    "scale_500_member": "Membership flag for the full 500-learner dataset",
    "audit_profile_model_use_allowed": "False flag preventing the audit profile from being used as a predictor",
    "diagnostic_correct_items": "Correct item count in the simulated baseline diagnostic.",
    "diagnostic_item_count": "Number of simulated diagnostic items",
    "diagnostic_score": "Correct diagnostic items divided by item count, expressed from 0 to 100.",
    "confidence_score": "Noisy topic confidence on a 1-to-5 scale; not a learning-style label.",
    "interaction_id": "Stable identifier for the selected learner-resource interaction.",
    "sequence_number": "Chronological interaction order within learner.",
    "simulated_day": "Synthetic day index used to preserve event chronology",
    "target_topic_id": "Topic addressed by the selected resource",
    "pre_score": "Latest observed topic score available before the interaction",
    "confidence_before": "Latest topic confidence available before the interaction",
    "completed": "Simulated binary resource-completion outcome",
    "time_spent_minutes": "Simulated time spent on the selected resource.",
    "attempt_count": "Simulated count of attempts or visits during the interaction",
    "usefulness_rating": "Optional simulated rating from 1 to 5 after completion",
    "post_correct_items": "Correct item count in the simulated post-resource check; missing when not observed",
    "post_item_count": "Number of post-resource items; missing when not observed",
    "post_score": "Simulated post-resource score from 0 to 100; missing when not observed",
    "score_gain": "Post score minus pre score; may be negative because assessments are noisy.",
    "outcome_observed": "1 when a post-resource score was generated and 0 otherwise",
    "chronological_split": "Within-learner historical training period or future holdout period.",
    "weakness_score": "One minus pre_score divided by 100",
    "prior_interaction_count": "Count of the learner's earlier interactions",
    "prior_completion_rate": "Completion rate over earlier interactions only",
    "prior_mean_observed_gain": "Mean score gain over earlier observed outcomes only; 0 when no history exists",
    "topic_prior_interaction_count": "Count of earlier interactions for the same topic",
    "topic_prior_mean_observed_gain": "Mean earlier observed gain for the same topic; 0 when unavailable",
    "format_prior_interaction_count": "Count of earlier interactions in the same broad format",
    "format_prior_completion_rate": "Earlier completion rate for the same broad format",
    "format_prior_mean_rating": "Mean earlier rating for the same format; 0 when unavailable",
    "cold_start_flag": "1 before the learner has three prior interactions",
    "prerequisite_count": "Count of direct prerequisites for the target topic",
    "prerequisite_min_score": "Minimum latest observed prerequisite score; 100 when no prerequisite exists",
    "prerequisite_mean_score": "Mean latest observed prerequisite score; 100 when no prerequisite exists",
    "unmet_prerequisite_count": "Direct prerequisites with latest observed score below 60",
    "resource_format_group": "Broad format of the selected resource",
    "resource_difficulty": "Researcher-assigned selected-resource difficulty from 1 to 3",
    "resource_duration_minutes": "Researcher-estimated selected-resource duration",
    "desired_resource_difficulty": "Difficulty band derived from the pre-interaction score",
    "difficulty_gap": "Absolute gap between resource difficulty and desired difficulty",
    "topic_curriculum_order": "Display order of the target topic",
    "topic_relative_difficulty": "Researcher-assigned target-topic difficulty",
    "target_completed": "Current interaction completion target; never a predictor for its own event",
    "target_score_gain": "Current interaction score-gain target; never a predictor for its own event",
    "target_usefulness_rating": "Current interaction rating target; never a predictor for its own event",
    "event_id": "Generator event identifier shared by all candidate counterfactuals",
    "candidate_resource_id": "Candidate resource considered for the event",
    "selected_resource_flag": "1 for the resource actually selected in the simulated log.",
    "oracle_completion_probability": "Generator-known completion probability; excluded from all model inputs.",
    "oracle_expected_mastery_gain": "Generator-known expected latent gain; excluded from all model inputs",
    "oracle_expected_gain_points": "Oracle expected latent gain multiplied by 100 for readability",
    "selection_probability_if_exploiting": "Softmax probability before the exploration override",
    "oracle_rank_within_event": "Rank of candidate expected gain within the event; 1 is best under simulator truth.",
    "model_use_allowed": "False flag identifying generator-only truth",
    "simulation_profile": "Generator-only stress profile",
    "general_ability": "Generator-only learner ability trait",
    "engagement_propensity": "Generator-only learner engagement trait",
    "learning_responsiveness": "Generator-only multiplier on latent mastery gain",
    "confidence_bias": "Generator-only bias used to make confidence imperfect",
    "individual_process_noise": "Generator-only learner-specific noise scale",
    "affinity_text": "Generator-only affinity for text resources",
    "affinity_practice": "Generator-only affinity for practice resources",
    "affinity_interactive": "Generator-only affinity for interactive resources",
    "affinity_video": "Generator-only affinity for video resources",
    "initial_latent_mastery": "Generator-only initial topic mastery",
    "latent_quality": "Generator-only resource quality",
    "latent_appeal": "Generator-only resource selection appeal",
    "selection_mode": "Whether selection used probabilistic fit or random exploration",
    "effective_selection_probability": "Probability associated with the realised selection mechanism",
    "latent_mastery_before": "Generator-only topic mastery before the interaction",
    "latent_mastery_after": "Generator-only topic mastery after the interaction",
    "latent_mastery_gain": "Generator-only non-negative mastery change",
    "latent_resource_quality": "Generator-only quality for the selected resource.",
    "latent_resource_appeal": "Generator-only appeal for the selected resource",
    "hidden_format_affinity": "Generator-only learner affinity for the selected format",
}


def build_data_dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Availability labels make the leakage boundary inspectable rather than implicit
    rows: list[dict[str, Any]] = []
    internal_tables = {
        "learner_truth",
        "learner_topic_truth",
        "resource_truth",
        "selected_interaction_truth",
        "candidate_potential_outcomes",
    }
    target_fields = {"completed", "score_gain", "usefulness_rating", "target_completed", "target_score_gain", "target_usefulness_rating"}
    identifier_fields = {
        "topic_id",
        "prerequisite_topic_id",
        "dependent_topic_id",
        "resource_id",
        "primary_topic_id",
        "learner_id",
        "interaction_id",
        "event_id",
        "candidate_resource_id",
        "target_topic_id",
    }
    for table_name, frame in frames.items():
        for field_name in frame.columns:
            description = FIELD_DESCRIPTIONS.get(field_name, field_name.replace("_", " ").capitalize() + ".")
            if table_name in internal_tables:
                role = "generator_truth"
                availability = "never_available_to_model"
                model_use = "exclude"
            elif field_name in target_fields or field_name.startswith("target_"):
                role = "outcome_or_target"
                availability = "after_interaction"
                model_use = "target_only"
            elif field_name in identifier_fields:
                role = "identifier_or_join_key"
                availability = "before_interaction_or_static"
                model_use = "split_or_join_only"
            elif field_name in {"audit_profile_group", "audit_profile_model_use_allowed"}:
                role = "audit_only"
                availability = "generator_setup"
                model_use = "exclude"
            else:
                role = "metadata_or_predictor"
                availability = "static_or_before_interaction"
                model_use = "eligible_subject_to_pipeline"
            rows.append(
                {
                    "table_name": table_name,
                    "field_name": field_name,
                    "pandas_dtype": str(frame[field_name].dtype),
                    "field_role": role,
                    "availability_time": availability,
                    "model_use": model_use,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset checks and audit figures
# ---------------------------------------------------------------------------

def validate_frames(frames: dict[str, pd.DataFrame], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Checks cover structure, ranges, leakage, sparsity and simulator behaviour
    topics = frames["topics"]
    prerequisites = frames["prerequisites"]
    resources = frames["resources"]
    learners = frames["learners"]
    baseline = frames["learner_topic_baseline"]
    interactions = frames["interactions"]
    model_ready = frames["model_ready"]
    selected_truth = frames["selected_interaction_truth"]
    potential = frames["candidate_potential_outcomes"]

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, observed: Any, expectation: str) -> None:
        # Store evidence as well as PASS or FAIL so a marker can inspect the reason
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expectation": expectation,
            }
        )

    edges = list(prerequisites[["prerequisite_topic_id", "dependent_topic_id"]].itertuples(index=False, name=None))
    try:
        topological_order(topics["topic_id"].tolist(), edges)
        graph_is_dag = True
    except ValueError:
        graph_is_dag = False
    add_check("prerequisite_graph_is_acyclic", graph_is_dag, graph_is_dag, "True")
    add_check("topic_count", len(topics) == 10, len(topics), "10")
    add_check("prerequisite_edge_count", len(prerequisites) == 12, len(prerequisites), "12")
    add_check("resource_count", len(resources) == 40, len(resources), "40")
    resource_counts = resources.groupby("primary_topic_id").size()
    add_check("four_resources_per_topic", bool((resource_counts == 4).all()), resource_counts.to_dict(), "Every topic has exactly 4 resources")
    add_check("learner_count", len(learners) == int(config["n_learners"]), len(learners), str(config["n_learners"]))
    scale_counts = {
        "scale_100_member": int(learners["scale_100_member"].sum()),
        "scale_250_member": int(learners["scale_250_member"].sum()),
        "scale_500_member": int(learners["scale_500_member"].sum()),
    }
    add_check(
        "learning_curve_subset_sizes",
        scale_counts == {"scale_100_member": 100, "scale_250_member": 250, "scale_500_member": len(learners)},
        scale_counts,
        "Nested subsets contain 100, 250 and all learners",
    )
    add_check("baseline_complete", len(baseline) == len(learners) * len(topics), len(baseline), "learner_count x topic_count")
    add_check("unique_interaction_ids", interactions["interaction_id"].is_unique, interactions["interaction_id"].nunique(), "All interaction IDs unique")
    add_check("foreign_key_learners", interactions["learner_id"].isin(learners["learner_id"]).all(), int((~interactions["learner_id"].isin(learners["learner_id"])).sum()), "0 orphan learner keys")
    add_check("foreign_key_resources", interactions["resource_id"].isin(resources["resource_id"]).all(), int((~interactions["resource_id"].isin(resources["resource_id"])).sum()), "0 orphan resource keys")
    sequence_ok = all(
        group["sequence_number"].tolist() == list(range(1, len(group) + 1))
        for _, group in interactions.sort_values(["learner_id", "sequence_number"]).groupby("learner_id")
    )
    add_check("chronological_sequence_complete", sequence_ok, sequence_ok, "Each learner sequence starts at 1 without gaps")
    score_columns = ["pre_score", "post_score"]
    score_ranges_ok = all(interactions[column].dropna().between(0, 100).all() for column in score_columns)
    add_check("score_ranges", score_ranges_ok, {column: [float(interactions[column].min()), float(interactions[column].max())] for column in score_columns}, "Scores between 0 and 100")
    add_check("confidence_range", interactions["confidence_before"].between(1, 5).all(), [float(interactions["confidence_before"].min()), float(interactions["confidence_before"].max())], "1 to 5")
    add_check("rating_range", interactions["usefulness_rating"].dropna().between(1, 5).all(), [float(interactions["usefulness_rating"].min()), float(interactions["usefulness_rating"].max())], "1 to 5 when observed")
    missing_post_policy = (
        interactions.loc[interactions["outcome_observed"] == 0, "post_score"].isna().all()
        and interactions.loc[interactions["outcome_observed"] == 1, "post_score"].notna().all()
    )
    add_check("post_outcome_missingness_policy", missing_post_policy, missing_post_policy, "Post score present exactly when outcome_observed=1")
    # These fields are useful for simulator evaluation but prohibited as predictors
    hidden_names = {
        "general_ability",
        "engagement_propensity",
        "learning_responsiveness",
        "confidence_bias",
        "latent_quality",
        "latent_appeal",
        "latent_mastery_before",
        "latent_mastery_after",
        "hidden_format_affinity",
        "oracle_expected_mastery_gain",
        "oracle_completion_probability",
    }
    hidden_overlap = sorted(hidden_names.intersection(model_ready.columns))
    add_check("no_generator_truth_in_model_ready", not hidden_overlap, hidden_overlap, "No hidden generator fields")
    prohibited_current_outcomes = {"completed", "time_spent_minutes", "attempt_count", "usefulness_rating", "post_score", "score_gain"}
    current_outcome_overlap = sorted(prohibited_current_outcomes.intersection(model_ready.columns))
    add_check("no_current_outcome_as_feature_name", not current_outcome_overlap, current_outcome_overlap, "Current outcomes appear only with target_ prefix")
    feature_columns = [
        column
        for column in model_ready.columns
        if column not in {"target_score_gain", "target_usefulness_rating"}
    ]
    missing_feature_count = int(model_ready[feature_columns].isna().sum().sum())
    add_check("no_missing_model_features", missing_feature_count == 0, missing_feature_count, "0 missing feature cells")
    add_check("latent_mastery_nondecreasing", 
              (selected_truth["latent_mastery_after"] >= selected_truth["latent_mastery_before"]).all(), 
              int((selected_truth["latent_mastery_after"] < selected_truth["latent_mastery_before"]).sum()), 
              "0 decreases")
    negative_observed_gains = int((interactions["score_gain"].dropna() < 0).sum())
    add_check("observed_negative_gains_exist", negative_observed_gains > 0, negative_observed_gains, ">0 because observed assessments are noisy")
    completion_rate = float(interactions["completed"].mean())
    add_check("completion_not_degenerate", 0.35 < completion_rate < 0.95, completion_rate, "Between 0.35 and 0.95")
    outcome_rate = float(interactions["outcome_observed"].mean())
    add_check("outcome_observation_not_degenerate", 0.25 < outcome_rate < 0.90, outcome_rate, "Between 0.25 and 0.90")
    interaction_counts = interactions.groupby("learner_id").size()
    add_check("interaction_budget_bounds", 
              interaction_counts.between(int(config["interaction_count"]["minimum"]), 
                                                                      int(config["interaction_count"]["maximum"])).all(), 
                                                                      [int(interaction_counts.min()), int(interaction_counts.max())], 
                                                                      "Within configured bounds")
    # Density uses observed learner-resource events over all possible pairs
    density = len(interactions) / (len(learners) * len(resources))
    add_check("interaction_matrix_is_sparse", density < 0.30, density, "Density below 0.30")
    correlation = float(
        frames["learner_topic_baseline"]
        .merge(frames["learner_topic_truth"], on=["learner_id", "topic_id"])[["diagnostic_score", "initial_latent_mastery"]]
        .corr()
        .iloc[0, 1]
    )
    add_check("diagnostic_is_informative_but_imperfect", 0.35 < correlation < 0.95, correlation, "Correlation between 0.35 and 0.95")
    candidate_counts = potential.groupby("event_id").size()
    add_check("four_candidate_outcomes_per_event", (candidate_counts == 4).all(), candidate_counts.value_counts().to_dict(), "4 candidates per event")
    add_check("model_ready_contains_observed_targets_only", model_ready["target_score_gain"].notna().all(), int(model_ready["target_score_gain"].isna().sum()), "0 missing score-gain targets")

    validation = pd.DataFrame(checks)
    summary = {
        "all_checks_passed": bool((validation["status"] == "PASS").all()),
        "check_count": int(len(validation)),
        "failed_check_count": int((validation["status"] == "FAIL").sum()),
        "n_topics": int(len(topics)),
        "n_resources": int(len(resources)),
        "n_learners": int(len(learners)),
        "n_interactions": int(len(interactions)),
        "n_observed_score_gain_targets": int(len(model_ready)),
        "matrix_density": density,
        "completion_rate": completion_rate,
        "outcome_observation_rate": outcome_rate,
        "mean_observed_score_gain": float(interactions["score_gain"].mean()),
        "median_observed_score_gain": float(interactions["score_gain"].median()),
        "negative_observed_gain_count": negative_observed_gains,
        "diagnostic_latent_mastery_correlation": correlation,
        "claim_boundary": config["claim_boundary"],
    }
    return validation, summary

# the write_plots function generates visualizations for auditing the synthetic dataset. 
# It creates histograms and scatter plots to illustrate distributions of diagnostic scores, 
# interaction counts, observed score gains, and audit profile counts. The plots are saved in 
# a validation directory for further inspection and review
def write_plots(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    # Figures are simple audit views rather than evidence of real learner behaviour
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline = frames["learner_topic_baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.arange(-2.5, 105, 12.5)
    ax.hist(baseline["diagnostic_score"], bins=bins, color="#2F6B7C", edgecolor="white")
    ax.set(title="Synthetic baseline diagnostic scores", xlabel="Diagnostic score", ylabel="Learner-topic records")
    fig.tight_layout()
    fig.savefig(validation_dir / "diagnostic_score_distribution.png", dpi=180)
    plt.close(fig)

    interactions = frames["interactions"]
    counts = interactions.groupby("learner_id").size()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(counts.value_counts().sort_index().index, counts.value_counts().sort_index().values, color="#4C78A8")
    ax.set(title="Interactions per synthetic learner", xlabel="Interaction count", ylabel="Learners")
    fig.tight_layout()
    fig.savefig(validation_dir / "interaction_count_distribution.png", dpi=180)
    plt.close(fig)

    observed = interactions[interactions["outcome_observed"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(observed["score_gain"], bins=np.arange(-65, 70, 10), color="#59A14F", edgecolor="white")
    axes[0].set(title="Observed score gain", xlabel="Post score minus pre score", ylabel="Interactions")
    axes[1].scatter(observed["pre_score"], observed["score_gain"], s=10, alpha=0.25, color="#E15759")
    axes[1].set(title="Ceiling effect audit", xlabel="Pre score", ylabel="Observed score gain")
    fig.tight_layout()
    fig.savefig(validation_dir / "outcome_audit.png", dpi=180)
    plt.close(fig)

    learners = frames["learners"]
    profile_counts = learners["audit_profile_group"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(profile_counts.index, profile_counts.values, color="#B279A2")
    ax.set(title="Programmatic stress profiles", xlabel="Synthetic learners", ylabel="Audit profile")
    fig.tight_layout()
    fig.savefig(validation_dir / "audit_profile_counts.png", dpi=180)
    plt.close(fig)

# here is a function that writes the generated dataset and associated validation outputs to disk. 
# It organizes the outputs into separate directories for public data, internal generator truth, 
# validation checks, and educator review templates. The function also generates a data dictionary, 
# validation report, and plots for auditing purposes
def write_outputs(frames: dict[str, pd.DataFrame], config: dict[str, Any], sources: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    # Keep model-facing data, simulator truth and validation evidence in separate folders
    data_dir = output_dir / "data"
    truth_dir = output_dir / "internal_generator_truth"
    validation_dir = output_dir / "validation"
    educator_dir = output_dir / "educator_validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    educator_dir.mkdir(parents=True, exist_ok=True)

    public_names = [
        "topics",
        "prerequisites",
        "resources",
        "resource_topics",
        "resource_topic_similarity",
        "learners",
        "learner_topic_baseline",
        "interactions",
        "model_ready",
        "model_ready_all_events",
    ]
    internal_names = [
        "learner_truth",
        "learner_topic_truth",
        "resource_truth",
        "selected_interaction_truth",
        "candidate_potential_outcomes",
    ]
    # Public tables may be used by the later candidate-building pipeline
    for name in public_names:
        frames[name].to_csv(data_dir / f"{name}.csv", index=False, float_format="%.8f")

    # Internal tables are retained for controlled evaluation only
    for name in internal_names:
        frames[name].to_csv(truth_dir / f"{name}.csv", index=False, float_format="%.8f")
    sources.to_csv(data_dir / "method_sources.csv", index=False)
    data_dictionary = build_data_dictionary({name: frames[name] for name in public_names + internal_names})
    data_dictionary.to_csv(data_dir / "data_dictionary.csv", index=False)

    # Blank review templates make later educator checking possible without changing raw data
    resource_review = frames["resources"][
        ["resource_id", "title", "url", "primary_topic_id", "difficulty", "duration_minutes", "format_group"]
    ].rename(
        columns={
            "primary_topic_id": "proposed_topic_id",
            "difficulty": "proposed_difficulty_1_to_3",
            "duration_minutes": "proposed_duration_minutes",
        }
    )
    for column in [
        "reviewer_id",
        "curriculum_relevance_1_to_5",
        "difficulty_agreement_yes_no",
        "suggested_difficulty_1_to_3",
        "duration_reasonable_yes_no",
        "topic_mapping_decision_accept_change",
        "suggested_topic_id",
        "access_issue_found",
        "review_comment",
        "review_date",
    ]:
        resource_review[column] = ""
    resource_review.to_csv(educator_dir / "resource_mapping_review_template.csv", index=False)

    edge_review = frames["prerequisites"].copy()
    for column in [
        "reviewer_id",
        "edge_decision_accept_reverse_remove",
        "confidence_1_to_5",
        "suggested_change",
        "review_comment",
        "review_date",
    ]:
        edge_review[column] = ""
    edge_review.to_csv(educator_dir / "prerequisite_edge_review_template.csv", index=False)

    # Save machine-readable checks and a short human-readable report
    validation, summary = validate_frames(frames, config)
    validation.to_csv(validation_dir / "validation_checks.csv", index=False)
    (validation_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_header = "| " + " | ".join(validation.columns) + " |"
    markdown_rule = "| " + " | ".join(["---"] * len(validation.columns)) + " |"
    markdown_rows = [
        "| "
        + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
        + " |"
        for row in validation.itertuples(index=False, name=None)
    ]
    report_lines = [
        "# Dataset validation report",
        "",
        f"All checks passed: **{summary['all_checks_passed']}**",
        "",
        f"- Learners: {summary['n_learners']:,}",
        f"- Interactions: {summary['n_interactions']:,}",
        f"- Observed score-gain targets: {summary['n_observed_score_gain_targets']:,}",
        f"- Learner-resource matrix density: {summary['matrix_density']:.3f}",
        f"- Completion rate: {summary['completion_rate']:.3f}",
        f"- Outcome observation rate: {summary['outcome_observation_rate']:.3f}",
        f"- Diagnostic/latent-mastery correlation: {summary['diagnostic_latent_mastery_correlation']:.3f}",
        "",
        "## Checks",
        "",
        markdown_header,
        markdown_rule,
        *markdown_rows,
        "",
        "## Claim boundary",
        "",
        config["claim_boundary"],
    ]
    (validation_dir / "validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_plots(frames, output_dir)

    # Hashes allow a later rerun to be compared with the submitted artefacts
    manifest_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.parts
        and "workbook_render" not in path.parts
        and path.suffix != ".pyc"
    )
    manifest = {
        "dataset_version": config["dataset_version"],
        "scenario": config["scenario"],
        "random_seed": config["random_seed"],
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in manifest_paths
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def prepare_output_directory(output_dir: Path, clean: bool) -> None:
    # Never recursively remove the source-code root even when --clean is supplied
    if clean and output_dir.exists() and output_dir.resolve() != ROOT.resolve():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

# Command-line entry point allows the dataset generator to be run as a script with 
# optional overrides for reproducibility and scale
def parse_args() -> argparse.Namespace:
    # Command-line overrides support scale and reproducibility checks without code edits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    parser.add_argument("--n-learners", type=int, default=None, help="Override the configured learner count.")
    parser.add_argument("--clean", action="store_true", help="Remove an external output directory before writing.")
    return parser.parse_args()


def main() -> None:
    # The notebook imports the functions above while this entry point supports direct use
    args = parse_args()
    config, topics, prerequisites, resources, sources = load_inputs(args.config)
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.n_learners is not None:
        config["n_learners"] = args.n_learners
    output_dir = args.output_dir.resolve()
    prepare_output_directory(output_dir, args.clean)
    frames = generate_dataset(config, topics, prerequisites, resources)
    summary = write_outputs(frames, config, sources, output_dir)
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit("Dataset generation completed, but one or more validation checks failed.")


if __name__ == "__main__":
    main()
