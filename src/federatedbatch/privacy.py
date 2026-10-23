"""Empirical embedding-leakage analyses used in the manuscript revision."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dense_values(adata, indices=None, features=None) -> np.ndarray:
    if indices is None and features is None:
        matrix = adata.X
    else:
        row_index = slice(None) if indices is None else indices
        column_index = slice(None) if features is None else features
        matrix = adata[row_index, column_index].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def fit_linear_decoder(
    auxiliary_embeddings: np.ndarray,
    auxiliary_expression: np.ndarray,
    *,
    alpha: float,
) -> Ridge:
    decoder = Ridge(alpha=alpha, fit_intercept=True)
    decoder.fit(auxiliary_embeddings, auxiliary_expression)
    return decoder


def _row_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    predicted_centered = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(observed_centered**2, axis=1) * np.sum(predicted_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _top_gene_overlap(observed: np.ndarray, predicted: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, observed.shape[1])
    observed_top = np.argpartition(observed, -top_k, axis=1)[:, -top_k:]
    predicted_top = np.argpartition(predicted, -top_k, axis=1)[:, -top_k:]
    overlaps = []
    for observed_row, predicted_row in zip(observed_top, predicted_top):
        intersection = len(np.intersect1d(observed_row, predicted_row, assume_unique=True))
        overlaps.append(intersection / (2 * top_k - intersection))
    return np.asarray(overlaps)


def reconstruction_metrics(
    decoder: Ridge,
    target_embeddings: np.ndarray,
    target_expression: np.ndarray,
    auxiliary_expression: np.ndarray,
    *,
    top_k: int = 50,
) -> tuple[dict[str, float], np.ndarray]:
    predicted = np.asarray(decoder.predict(target_embeddings), dtype=np.float32)
    baseline = np.broadcast_to(auxiliary_expression.mean(axis=0), target_expression.shape)
    mse = mean_squared_error(target_expression, predicted)
    baseline_mse = mean_squared_error(target_expression, baseline)
    correlations = _row_pearson(target_expression, predicted)
    overlaps = _top_gene_overlap(target_expression, predicted, top_k)
    metrics = {
        "n_targets": int(target_expression.shape[0]),
        "n_genes": int(target_expression.shape[1]),
        "mse": float(mse),
        "mean_expression_baseline_mse": float(baseline_mse),
        "relative_mse": float(mse / baseline_mse) if baseline_mse > 0 else np.nan,
        "variance_weighted_r2": float(
            r2_score(target_expression, predicted, multioutput="variance_weighted")
        ),
        "mean_cell_pearson": float(np.nanmean(correlations)),
        "median_cell_pearson": float(np.nanmedian(correlations)),
        "mean_top_gene_jaccard": float(np.mean(overlaps)),
    }
    return metrics, predicted


def common_membership_features(
    embeddings: np.ndarray,
    expression: np.ndarray,
    decoder: Ridge,
) -> np.ndarray:
    reconstructed = np.asarray(decoder.predict(embeddings), dtype=np.float32)
    residual = np.mean((expression - reconstructed) ** 2, axis=1)
    return np.column_stack(
        [
            np.linalg.norm(embeddings, axis=1),
            residual,
            np.linalg.norm(reconstructed, axis=1),
        ]
    )


def queue_proximity_features(
    embeddings: np.ndarray,
    queue_embeddings: np.ndarray,
    *,
    chunk_size: int = 2048,
    top_k: int = 5,
) -> np.ndarray:
    if not len(queue_embeddings):
        raise ValueError("queue proximity features require a non-empty queue")
    queue_norm = queue_embeddings / np.maximum(
        np.linalg.norm(queue_embeddings, axis=1, keepdims=True), 1e-12
    )
    outputs = []
    top_k = min(top_k, len(queue_embeddings))
    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        chunk_norm = chunk / np.maximum(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
        similarities = chunk_norm @ queue_norm.T
        nearest_cosine = similarities.max(axis=1)
        top_similarities = np.partition(similarities, -top_k, axis=1)[:, -top_k:]
        mean_top_cosine = top_similarities.mean(axis=1)
        outputs.append(np.column_stack([nearest_cosine, mean_top_cosine]))
    return np.concatenate(outputs, axis=0)


def membership_attack(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    *,
    seed: int,
    test_fraction: float = 0.5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_per_class = min(len(member_features), len(nonmember_features))
    member_indices = rng.choice(len(member_features), n_per_class, replace=False)
    nonmember_indices = rng.choice(len(nonmember_features), n_per_class, replace=False)
    features = np.vstack(
        [member_features[member_indices], nonmember_features[nonmember_indices]]
    )
    labels = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    classifier = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    classifier.fit(train_x, train_y)
    probabilities = classifier.predict_proba(test_x)[:, 1]
    predictions = probabilities >= 0.5
    auc = float(roc_auc_score(test_y, probabilities))
    return {
        "n_candidates_per_class": int(n_per_class),
        "n_attack_test": int(len(test_y)),
        "roc_auc": auc,
        "attack_advantage": float(2 * auc - 1),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
    }


def queue_staleness(
    stored_embeddings: np.ndarray,
    final_embeddings: np.ndarray,
    records: list[dict],
    *,
    final_round: int,
) -> pd.DataFrame:
    if len(stored_embeddings) != len(final_embeddings) or len(records) != len(stored_embeddings):
        raise ValueError("queue staleness inputs must have equal lengths")
    stored_norm = stored_embeddings / np.maximum(
        np.linalg.norm(stored_embeddings, axis=1, keepdims=True), 1e-12
    )
    final_norm = final_embeddings / np.maximum(
        np.linalg.norm(final_embeddings, axis=1, keepdims=True), 1e-12
    )
    similarities = np.sum(stored_norm * final_norm, axis=1)
    rows = []
    for record, similarity in zip(records, similarities):
        rows.append(
            {
                **record,
                "age_rounds": final_round - int(record["collected_round"]),
                "cosine_similarity_to_final": float(similarity),
                "cosine_drift": float(1 - similarity),
            }
        )
    return pd.DataFrame(rows)
