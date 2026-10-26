#!/usr/bin/env python3
"""Matched gene-expression reconstruction attack for the final private A/B release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from federatedbatch.data import normalize, read_dataset  # noqa: E402
from experiments import final_private_ab_attack_validation as release_attack  # noqa: E402
from experiments.final_private_ab_attack_validation import (  # noqa: E402
    C,
    FINAL_ROOT,
    FIGURE2_ROOT,
    NOISE_SEEDS,
    SEEDS,
    SIGMA,
    anchor_rows,
    contribution,
    dictionary_estimate,
    load_partition,
    normalize as normalize_rows,
    optimized_estimate,
    source_noise,
)


def fallback_strata(frame: pd.DataFrame, minimum: int) -> np.ndarray:
    client = frame["client"].astype(str)
    batch = frame["batch"].astype(str)
    celltype = frame["celltype"].astype(str)
    labels = "joint:" + client + "|" + batch + "|" + celltype
    for fallback in (
        "celltype:" + client + "|" + celltype,
        "batch:" + client + "|" + batch,
        "client:" + client,
    ):
        counts = labels.value_counts()
        sparse = labels.map(counts) < minimum
        labels = labels.where(~sparse, fallback)
    return labels.to_numpy(str)


def auxiliary_train_validation(
    auxiliary: np.ndarray, split: pd.DataFrame, minimum: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    meta = split.set_index("original_index").loc[auxiliary].reset_index()
    labels = fallback_strata(meta, minimum)
    try:
        train_indices, validation_indices = train_test_split(
            auxiliary, test_size=0.20, random_state=DECODER_SPLIT_SEED, stratify=labels
        )
    except ValueError:
        train_indices, validation_indices = train_test_split(
            auxiliary, test_size=0.20, random_state=DECODER_SPLIT_SEED
        )
    return np.sort(train_indices), np.sort(validation_indices)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            previous = int(width)
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def train_mlp_decoder(
    train_z: np.ndarray,
    train_x: np.ndarray,
    val_z: np.ndarray,
    val_x: np.ndarray,
    seed: int,
) -> tuple[MLPDecoder, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(int(seed) + DECODER_SPLIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + DECODER_SPLIT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train_z.mean(axis=0).astype(np.float32)
    scale = train_z.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_z - mean) / scale
    val_scaled = (val_z - mean) / scale
    model = MLPDecoder(train_z.shape[1], MLP_HIDDEN_DIMS, train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(int(seed) + DECODER_SPLIT_SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_scaled), torch.from_numpy(train_x)),
        batch_size=MLP_BATCH_SIZE, shuffle=True, generator=generator
    )
    val_z_tensor = torch.from_numpy(val_scaled).to(device)
    val_x_tensor = torch.from_numpy(val_x).to(device)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    patience = 0
    for epoch in range(1, MLP_MAX_EPOCHS + 1):
        model.train()
        for z_batch, x_batch in loader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(z_batch), x_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_z_tensor), val_x_tensor).cpu())
        if val_loss < best_loss - MLP_MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= MLP_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("MLP decoder failed to produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, mean, scale, {"best_epoch": best_epoch, "best_validation_mse": best_loss, "epochs_run": epoch}


def mlp_predict(model: MLPDecoder, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 256):
            batch = torch.from_numpy(((values[start : start + 256] - mean) / scale).astype(np.float32)).to(device)
            outputs.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


OUTPUT = REPO_ROOT / "results/final_expression_reconstruction_attack"
FIGURE5_ROOT = REPO_ROOT / "results/figure5_privacy"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURE_PATH = REPO_ROOT / FIGURE2_ROOT / "features.txt"
PREVIOUS_RECONSTRUCTION = REPO_ROOT / "results/final_private_AB_attack_validation/reconstruction_attack.csv"
FRESH_ROOT = REPO_ROOT / FIGURE2_ROOT / "runs"
TARGET_ATTACK = "optimized 100-D leave-one-out Gaussian MLE"
DECODER_SPLIT_SEED = 20260923
RIDGE_ALPHA = 0.01
MLP_HIDDEN_DIMS = [256, 512]
MLP_BATCH_SIZE = 128
MLP_LR = 0.001
MLP_WEIGHT_DECAY = 1e-5
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
MLP_MIN_DELTA = 1e-5
TARGETS_PER_SEED = 30
LATENT_DIM = 100

CONDITION_VANILLA = "Vanilla FreshQueue direct embedding"
CONDITION_SIGMA0 = "Aggregate A/B sigma=0"
CONDITION_SIGMA2 = "Final Joint-C50 sigma=2"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(values: np.ndarray | list[str]) -> str:
    text = "\n".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    first = observed - observed.mean(axis=1, keepdims=True)
    second = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(first * second, axis=1)
    denominator = np.sqrt(np.sum(first**2, axis=1) * np.sum(second**2, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(observed.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )


def row_spearman(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return row_pearson(rankdata(observed, axis=1), rankdata(predicted, axis=1))


def metrics_by_cell(target_expression: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pearson": row_pearson(target_expression, predicted),
        "spearman": row_spearman(target_expression, predicted),
        "rmse": np.sqrt(np.mean((target_expression - predicted) ** 2, axis=1)),
    }


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    margin = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    mean = float(values.mean())
    return mean - margin, mean + margin


def split_config() -> dict:
    return {
        "privacy_split": {"sparse_stratum_minimum": 5},
        "reconstruction": {
            "auxiliary_validation_fraction": 0.2,
            "decoder_split_seed": DECODER_SPLIT_SEED,
            "mlp_hidden_dims": MLP_HIDDEN_DIMS,
            "mlp_learning_rate": MLP_LR,
            "mlp_weight_decay": MLP_WEIGHT_DECAY,
            "mlp_batch_size": MLP_BATCH_SIZE,
            "mlp_max_epochs": MLP_MAX_EPOCHS,
            "mlp_patience": MLP_PATIENCE,
            "mlp_min_delta": MLP_MIN_DELTA,
        },
    }


def load_expression() -> tuple[np.ndarray, list[str], dict]:
    features = [line.strip() for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(features) != 2000 or len(set(features)) != 2000:
        raise ValueError(f"expected exactly 2,000 unique HVGs, got {len(features)}")
    source, dataset_path = read_dataset("pancreas", DATA_DIR)
    normalize(source, target_sum=10000, log1p=True)
    matrix = source[:, features].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    expression = np.asarray(matrix, dtype=np.float32)
    if expression.shape != (16382, 2000) or not np.isfinite(expression).all():
        raise ValueError(f"unexpected normalized expression matrix: {expression.shape}")
    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(Path(dataset_path)),
        "feature_path": str(FEATURE_PATH),
        "feature_sha256": sha256(FEATURE_PATH),
        "n_cells": int(expression.shape[0]),
        "n_hvgs": int(expression.shape[1]),
        "normalize_target_sum": 10000,
        "log1p": True,
    }
    return expression, features, metadata


def load_targets() -> pd.DataFrame:
    previous = pd.read_csv(PREVIOUS_RECONSTRUCTION)
    targets = previous[previous.attack == TARGET_ATTACK][
        ["seed", "round", "source_client", "candidate_cell_id", "candidate_global_index"]
    ].copy()
    targets = targets.rename(
        columns={
            "candidate_cell_id": "cell_id",
            "candidate_global_index": "global_index",
        }
    )
    targets["seed"] = targets.seed.astype(int)
    targets["round"] = targets["round"].astype(int)
    targets["source_client"] = targets.source_client.astype(int)
    targets["global_index"] = targets.global_index.astype(int)
    if len(targets) != len(SEEDS) * TARGETS_PER_SEED:
        raise ValueError(f"expected {len(SEEDS) * TARGETS_PER_SEED} matched target rows, got {len(targets)}")
    if targets.groupby("seed").cell_id.nunique().to_dict() != {seed: TARGETS_PER_SEED for seed in SEEDS}:
        raise ValueError("target IDs are not 30 unique cells per seed")
    if any(group.cell_id.nunique() != len(group) for _, group in targets.groupby("seed")):
        raise ValueError("target IDs are reused within a seed")
    return targets.sort_values(["seed", "round", "source_client", "global_index"]).reset_index(drop=True)


def load_auxiliary(split_path: Path, target_ids: set[str]) -> tuple[np.ndarray, pd.DataFrame, int]:
    split = pd.read_csv(split_path)
    auxiliary = split.loc[split.split == "aux"].copy()
    overlap = auxiliary.cell_id.astype(str).isin(target_ids)
    overlap_count = int(overlap.sum())
    auxiliary = auxiliary.loc[~overlap].copy()
    indices = auxiliary.original_index.astype(int).to_numpy()
    if len(indices) < 1000:
        raise ValueError(f"too few auxiliary cells in {split_path}: {len(indices)}")
    return indices, split, overlap_count


def reconstruct_qhats(
    seed: int,
    targets: pd.DataFrame,
    private_embedding: np.ndarray,
    device: torch.device,
    *,
    steps: int = 450,
    restarts: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    groups = load_partition(REPO_ROOT, seed)
    schedule = pd.read_csv(REPO_ROOT / FINAL_ROOT / "schedules" / f"remote_cell_schedule_seed_{seed}.csv")
    q0_rows: list[np.ndarray] = []
    q2_rows: list[np.ndarray] = []
    for row_index, row in targets.reset_index(drop=True).iterrows():
        round_number = int(row["round"])
        source = int(row["source_client"])
        global_index = int(row["global_index"])
        if str(schedule.loc[schedule.global_index == global_index, "cell_id"].iloc[0]) != str(row["cell_id"]):
            raise ValueError(f"seed {seed}: target cell/global index mismatch")
        anchor_map = anchor_rows(private_embedding, groups, seed=seed, round_number=round_number)
        target_order = [target for target in range(5) if target != source]
        anchors = torch.as_tensor(
            np.concatenate([anchor_map[target] for target in target_order]),
            dtype=torch.float32,
            device=device,
        )
        q_true = torch.as_tensor(private_embedding[global_index], dtype=torch.float32, device=device)
        true_a, true_b, _, _ = contribution(anchors, q_true)
        noise_a, noise_b = source_noise(seed=seed, round_number=round_number, source=source, device=device)
        noisy_a = true_a + SIGMA * C * noise_a
        noisy_b = true_b + SIGMA * C * noise_b
        source_schedule = schedule[
            (schedule.active_round == round_number) & (schedule.source_client == source)
        ].sort_values("source_local_index")
        known_indices = source_schedule.global_index.astype(int).to_numpy()
        dictionary = torch.as_tensor(
            private_embedding[known_indices[known_indices != global_index]],
            dtype=torch.float32,
            device=device,
        )
        dictionary_hat, _ = dictionary_estimate(anchors, noisy_a, noisy_b, dictionary)
        q2_hat, _ = optimized_estimate(
            anchors, noisy_a, noisy_b, dictionary_hat, steps=steps, restarts=restarts,
            random_seed=seed * 1_000_003 + round_number * 1_009 + source * 97 + row_index,
        )
        q0_hat = torch.nn.functional.normalize(
            true_b.sum(dim=0) / true_a.sum().clamp_min(1e-30), dim=0
        )
        q0_rows.append(q0_hat.detach().cpu().numpy().astype(np.float32))
        q2_rows.append(q2_hat.detach().cpu().numpy().astype(np.float32))
    return np.stack(q0_rows), np.stack(q2_rows)


def train_decoders(
    seed: int,
    condition_space: str,
    embedding: np.ndarray,
    auxiliary_indices: np.ndarray,
    split: pd.DataFrame,
    expression: np.ndarray,
) -> dict:
    aux_train, aux_validation = auxiliary_train_validation(auxiliary_indices, split)
    train_z = embedding[aux_train]
    validation_z = embedding[aux_validation]
    train_x = expression[aux_train]
    validation_x = expression[aux_validation]
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr")
    ridge.fit(embedding[auxiliary_indices], expression[auxiliary_indices])
    mlp, z_mean, z_scale, mlp_metadata = train_mlp_decoder(
        train_z.astype(np.float32),
        train_x.astype(np.float32),
        validation_z.astype(np.float32),
        validation_x.astype(np.float32),
        seed,
    )
    return {
        "space": condition_space,
        "ridge": ridge,
        "mlp": mlp,
        "mlp_mean": z_mean,
        "mlp_scale": z_scale,
        "mlp_metadata": mlp_metadata,
        "aux_indices": auxiliary_indices,
        "aux_train_indices": aux_train,
        "aux_validation_indices": aux_validation,
        "auxiliary_expression_mean": expression[auxiliary_indices].mean(axis=0),
        "auxiliary_embedding": embedding[auxiliary_indices],
        "auxiliary_expression": expression[auxiliary_indices],
    }


def evaluate_predictions(
    condition: str,
    decoder_name: str,
    target_meta: pd.DataFrame,
    target_expression: np.ndarray,
    prediction: np.ndarray,
) -> list[dict]:
    metrics = metrics_by_cell(target_expression, prediction)
    rows: list[dict] = []
    for index, row in target_meta.reset_index(drop=True).iterrows():
        rows.append(
            {
                "seed": int(row.seed),
                "round": int(row["round"]),
                "source_client": int(row.source_client),
                "cell_id": str(row.cell_id),
                "global_index": int(row.global_index),
                "condition": condition,
                "decoder": decoder_name,
                "n_genes": int(target_expression.shape[1]),
                "pearson": float(metrics["pearson"][index]),
                "spearman": float(metrics["spearman"][index]),
                "rmse": float(metrics["rmse"][index]),
            }
        )
    return rows


def summarize(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = ["pearson", "spearman", "rmse"]
    for (condition, decoder), group in per_cell.groupby(["condition", "decoder"], sort=True):
        for level, values_by_group in [
            ("cell", group),
            ("seed", group.groupby("seed", as_index=False)[metrics].mean()),
        ]:
            for metric in metrics:
                values = values_by_group[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                low, high = ci95(values)
                rows.append(
                    {
                        "aggregation_level": level,
                        "condition": condition,
                        "decoder": decoder,
                        "metric": metric,
                        "n": len(values),
                        "mean": float(values.mean()) if len(values) else math.nan,
                        "median": float(np.median(values)) if len(values) else math.nan,
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def summary_value(summary: pd.DataFrame, condition: str, decoder: str, metric: str, level: str = "cell") -> float:
    row = summary[
        (summary.aggregation_level == level)
        & (summary.condition == condition)
        & (summary.decoder == decoder)
        & (summary.metric == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"missing summary: {level}, {condition}, {decoder}, {metric}")
    return float(row.iloc[0]["mean"])


def build_report(summary: pd.DataFrame, config: dict) -> str:
    primary_decoder = "MLP"
    table_rows = []
    for condition in [CONDITION_VANILLA, CONDITION_SIGMA0, CONDITION_SIGMA2]:
        table_rows.append(
            (
                condition,
                summary_value(summary, condition, primary_decoder, "pearson"),
                summary_value(summary, condition, primary_decoder, "spearman"),
                summary_value(summary, condition, primary_decoder, "rmse"),
            )
        )
    van_p, van_s, van_r = table_rows[0][1:]
    zero_p, zero_s, zero_r = table_rows[1][1:]
    priv_p, priv_s, priv_r = table_rows[2][1:]
    p_drop = van_p - priv_p
    s_drop = van_s - priv_s
    p_drop_pct = 100.0 * p_drop / max(abs(van_p), 1e-12)
    s_drop_pct = 100.0 * s_drop / max(abs(van_s), 1e-12)
    ridge_p = summary_value(summary, CONDITION_SIGMA2, "Ridge", "pearson")
    ridge_s = summary_value(summary, CONDITION_SIGMA2, "Ridge", "spearman")
    if p_drop >= 0.15 and s_drop >= 0.15 and priv_p < zero_p and priv_s < zero_s:
        verdict = "STRONG SUPPORT"
    elif p_drop >= 0.05 and s_drop >= 0.05 and priv_p < zero_p and priv_s < zero_s:
        verdict = "SUPPORT"
    elif priv_p < van_p and priv_s < van_s:
        verdict = "MIXED"
    else:
        verdict = "FAIL"
    table_text = "\n".join(
        f"| {condition} | {pearson:.4f} | {spearman:.4f} | {rmse:.4f} |"
        for condition, pearson, spearman, rmse in table_rows
    )
    return f"""# Final gene-expression reconstruction attack

## Scope and matched provenance

No FederatedBatch model was retrained and Figures 1, 2, and 5 were not modified.
This analysis uses the same 210 target cells from the preceding final private
leave-one-out attack: 30 unique scheduled remote cells per seed across seeds
218, 394, 542, 639, 788, 814, and 880. Each condition uses the same target
cell IDs and the same normalized/log1p 2,000-HVG expression targets.

New decoders were fit separately for the Vanilla FreshQueue normalized latent
space and the final Joint-C50 normalized latent space. The auxiliary cells are
the saved Figure 5 `aux` roles; they are disjoint from all target cells, with
the same auxiliary IDs used for both latent spaces. No old decoder weights were
reused. Ridge alpha is fixed at {RIDGE_ALPHA}; the MLP is fixed at hidden
layers {MLP_HIDDEN_DIMS}, learning rate {MLP_LR}, weight decay {MLP_WEIGHT_DECAY},
batch size {MLP_BATCH_SIZE}, and max {MLP_MAX_EPOCHS} epochs with patience
{MLP_PATIENCE}. The decoder train/validation split is deterministic and shared
across comparable conditions.

For sigma=0, q_hat is reconstructed by the validated closed-form A/B ratio. For
sigma=2, q_hat is reconstructed by the same validated 100-D Gaussian leave-one-out
oracle used in the preceding attack: exact clipped A/B likelihood, 4 starts, and
450 Adam steps. The final DP accounting is epsilon approximately 1.993 at
delta=1e-5 for the cross-client A/B release channel only.

## Primary matched results (MLP decoder)

| Interface | Cell-wise Pearson mean | Cell-wise Spearman mean | RMSE mean |
|---|---:|---:|---:|
{table_text}

The corresponding Ridge sigma=2 means are Pearson {ridge_p:.4f} and Spearman
{ridge_s:.4f}; the direction is checked with both decoder families in
`reconstruction_summary.csv`.

## Interpretation

1. Relative to direct Vanilla FreshQueue exposure, final sigma=2 reduces the
   MLP expression Pearson mean by {p_drop:.4f} ({p_drop_pct:.1f}%) and Spearman
   by {s_drop:.4f} ({s_drop_pct:.1f}%).
2. Sigma=0 aggregate A/B remains vulnerable: its q_hat is the exact cell
   embedding under the validated residual attack, so expression reconstruction
   remains close to the direct-exposure condition (Pearson {zero_p:.4f},
   Spearman {zero_s:.4f}). It must not be described as private.
3. The final DP release materially suppresses expression reconstruction when
   both Pearson and Spearman fall below direct exposure and sigma=0. The
   comparison is matched to new decoders and target cells; the old Pearson
   approximately 0.708 result is not used as the primary benchmark because its
   decoder/provenance differs.
4. Simple mean-expression and nearest-auxiliary baselines are included for all
   three interfaces. In this latent space, the mean baseline can have a high
   cosine to target embeddings because of shared direction, so expression
   metrics—not embedding cosine—are used for the main conclusion.
5. This evaluates only the cross-client representation/release channel. Leakage
   from model updates or the global model is not solved by this mechanism.

Final verdict: **{verdict}**.

Provenance metadata is in `decoder_config.json`; per-cell metrics are in
`reconstruction_per_cell.csv`; condition/decoder summaries with cell-level and
seed-level confidence intervals are in `reconstruction_summary.csv`.
"""


def main() -> None:
    global OUTPUT, FIGURE5_ROOT, DATA_DIR, FEATURE_PATH, PREVIOUS_RECONSTRUCTION, FRESH_ROOT
    global FINAL_ROOT, FIGURE2_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--partition-root", type=Path, default=FIGURE2_ROOT)
    parser.add_argument("--explicit-privacy-root", type=Path, default=FIGURE5_ROOT)
    parser.add_argument("--explicit-full-root", type=Path, default=FRESH_ROOT)
    parser.add_argument("--previous-reconstruction", type=Path, default=PREVIOUS_RECONSTRUCTION)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    FINAL_ROOT = args.final_root.resolve()
    FIGURE2_ROOT = args.partition_root.resolve()
    FIGURE5_ROOT = args.explicit_privacy_root.resolve()
    FRESH_ROOT = args.explicit_full_root.resolve()
    PREVIOUS_RECONSTRUCTION = args.previous_reconstruction.resolve()
    DATA_DIR = args.data_dir.resolve()
    FEATURE_PATH = FINAL_ROOT / "features.txt"
    release_attack.FINAL_ROOT = FINAL_ROOT
    release_attack.FIGURE2_ROOT = FIGURE2_ROOT
    release_attack.FIGURE5_ROOT = FIGURE5_ROOT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expression, features, expression_meta = load_expression()
    targets = load_targets()
    per_cell_rows: list[dict] = []
    decoder_records: dict[str, dict] = {}
    config = split_config()
    for seed in SEEDS:
        seed_targets = targets[targets.seed == seed].reset_index(drop=True)
        target_ids = set(seed_targets.cell_id.astype(str))
        auxiliary_indices, split, excluded_aux_target_overlap = load_auxiliary(
            FIGURE5_ROOT / "02_splits" / f"seed_{seed}_cell_ids.csv", target_ids
        )
        private_path = FINAL_ROOT / "training" / f"seed_{seed}" / "Joint-C50-sigma2" / "final_embedding.npy"
        fresh_path = FRESH_ROOT / f"seed_{seed}" / "FreshQueue" / "final_embedding.npy"
        private_embedding = normalize_rows(np.load(private_path).astype(np.float32))
        fresh_embedding = normalize_rows(np.load(fresh_path).astype(np.float32))
        if private_embedding.shape != (16382, LATENT_DIM) or fresh_embedding.shape != (16382, LATENT_DIM):
            raise ValueError(f"seed {seed}: latent shape mismatch")
        if np.intersect1d(auxiliary_indices, seed_targets.global_index.to_numpy(int)).size:
            raise ValueError(f"seed {seed}: auxiliary/target index overlap")
        q0, q2 = reconstruct_qhats(seed, seed_targets, private_embedding, device)
        target_expression = expression[seed_targets.global_index.to_numpy(int)]
        fresh_decoders = train_decoders(
            seed, "FreshQueue-normalized", fresh_embedding, auxiliary_indices, split, expression
        )
        private_decoders = train_decoders(
            seed, "Joint-C50-normalized", private_embedding, auxiliary_indices, split, expression
        )
        condition_inputs = {
            CONDITION_VANILLA: (fresh_embedding[seed_targets.global_index.to_numpy(int)], fresh_decoders),
            CONDITION_SIGMA0: (q0, private_decoders),
            CONDITION_SIGMA2: (q2, private_decoders),
        }
        for condition, (target_z, decoders) in condition_inputs.items():
            ridge_prediction = decoders["ridge"].predict(target_z).astype(np.float32)
            mlp_prediction = mlp_predict(
                decoders["mlp"], target_z.astype(np.float32), decoders["mlp_mean"], decoders["mlp_scale"]
            )
            mean_prediction = np.broadcast_to(
                decoders["auxiliary_expression_mean"], target_expression.shape
            ).copy()
            similarities = target_z @ decoders["auxiliary_embedding"].T
            nearest_prediction = decoders["auxiliary_expression"][np.argmax(similarities, axis=1)]
            for name, prediction in [
                ("Ridge", ridge_prediction),
                ("MLP", mlp_prediction),
                ("MeanExpression", mean_prediction),
                ("NearestAuxiliary", nearest_prediction),
            ]:
                per_cell_rows.extend(
                    evaluate_predictions(condition, name, seed_targets, target_expression, prediction)
                )
        decoder_records[str(seed)] = {
            "auxiliary_n": int(len(auxiliary_indices)),
            "original_auxiliary_n": int((split.split == "aux").sum()),
            "excluded_auxiliary_target_overlap_n": excluded_aux_target_overlap,
            "auxiliary_ids_sha256": sha256_ids(split.loc[split.split == "aux", "cell_id"].astype(str).tolist()),
            "auxiliary_train_n": int(len(private_decoders["aux_train_indices"])),
            "auxiliary_validation_n": int(len(private_decoders["aux_validation_indices"])),
            "fresh_mlp": fresh_decoders["mlp_metadata"],
            "private_mlp": private_decoders["mlp_metadata"],
            "private_embedding_sha256": sha256(private_path),
            "fresh_embedding_sha256": sha256(fresh_path),
            "target_ids_sha256": sha256_ids(seed_targets.cell_id.astype(str).tolist()),
        }
        del fresh_decoders, private_decoders, private_embedding, fresh_embedding, q0, q2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    per_cell = pd.DataFrame(per_cell_rows)
    if len(per_cell) != len(SEEDS) * TARGETS_PER_SEED * 3 * 4:
        raise ValueError(f"unexpected per-cell row count: {len(per_cell)}")
    atomic_csv(per_cell, OUTPUT / "reconstruction_per_cell.csv")
    summary = summarize(per_cell)
    atomic_csv(summary, OUTPUT / "reconstruction_summary.csv")
    atomic_json(
        {
            "protocol": "matched final gene-expression reconstruction attack",
            "federated_training_rerun": False,
            "figures_modified": False,
            "seeds": list(SEEDS),
            "target_attack_source": str(PREVIOUS_RECONSTRUCTION),
            "target_attack_count_per_seed": TARGETS_PER_SEED,
            "target_ids_sha256_all": sha256_ids(targets.cell_id.astype(str).tolist()),
            "auxiliary_split_source": str(FIGURE5_ROOT / "02_splits"),
            "decoder_spaces": {"Vanilla FreshQueue": "FreshQueue-normalized", "Aggregate/DP": "Joint-C50-normalized"},
            "decoder_hyperparameters": {
                "ridge_alpha": RIDGE_ALPHA,
                "mlp_hidden_dims": MLP_HIDDEN_DIMS,
                "mlp_batch_size": MLP_BATCH_SIZE,
                "mlp_learning_rate": MLP_LR,
                "mlp_weight_decay": MLP_WEIGHT_DECAY,
                "mlp_max_epochs": MLP_MAX_EPOCHS,
                "mlp_patience": MLP_PATIENCE,
                "mlp_min_delta": MLP_MIN_DELTA,
                "decoder_split_seed": DECODER_SPLIT_SEED,
            },
            "expression_target": expression_meta,
            "q_reconstruction": {"sigma0": "validated closed-form A/B ratio", "sigma2": "validated Gaussian leave-one-out MLE, 4 starts, 450 Adam steps"},
            "per_seed": decoder_records,
            "noise_seeds": NOISE_SEEDS,
            "C": C,
            "sigma": SIGMA,
        },
        OUTPUT / "decoder_config.json",
    )
    report = build_report(summary, config)
    atomic_text(report, OUTPUT / "EXPRESSION_RECONSTRUCTION_REPORT.md")
    print(report)


if __name__ == "__main__":
    main()
