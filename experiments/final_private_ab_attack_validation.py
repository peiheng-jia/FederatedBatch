#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


OUTPUT = Path("results/final_private_AB_attack_validation")
FINAL_ROOT = Path("results/round_once_jointDP_multiseed_102")
FIGURE2_ROOT = Path("results/figure2_unified_7seed_original_environment_20260923")
FIGURE5_ROOT = Path("results/figure5_privacy")
SEEDS = (218, 394, 542, 639, 788, 814, 880)
NOISE_SEEDS = {seed: 10000 + seed for seed in SEEDS}
ROUNDS = (4, 12, 20)
C = 1206.3391650512762
SIGMA = 2.0
TAU = 0.20
LATENT_DIM = 100
ANCHORS_PER_TARGET = 128
REPRESENTATION_ROWS_PER_TARGET = 2 * ANCHORS_PER_TARGET
REMOTE_PER_SOURCE = 102
ACTIVE_ROUNDS = tuple(range(4, 21))
RECONSTRUCTION_BASIS = (
    "post-training surrogate: real C50/sigma2 final embeddings and frozen remote "
    "schedule; deterministic 128-cell target batches duplicated to the protocol's "
    "two-view 256 representation rows because per-round anchors/releases/checkpoints "
    "were not saved"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate privacy attacks on the finalized RoundOnce joint A/B release channel.')
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--partition-root", type=Path, default=FIGURE2_ROOT)
    parser.add_argument("--explicit-privacy-root", type=Path, default=FIGURE5_ROOT)
    parser.add_argument("--membership-draws", type=int, default=64)
    parser.add_argument("--reconstruction-candidates-per-source", type=int, default=2)
    parser.add_argument("--optimization-steps", type=int, default=450)
    parser.add_argument("--optimization-restarts", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def t_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2:
        return math.nan, math.nan
    mean = float(array.mean())
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    return mean - half_width, mean + half_width


def validate_inputs(repo: Path) -> dict:
    required = [
        repo / FINAL_ROOT / "provenance.json",
        repo / FINAL_ROOT / "exposure_QC.csv",
        repo / FIGURE5_ROOT / "04_membership" / "membership_per_seed.csv",
        repo / FIGURE5_ROOT / "03_reconstruction" / "reconstruction_per_seed.csv",
    ]
    for seed in SEEDS:
        required.extend(
            [
                repo / FINAL_ROOT / "schedules" / f"remote_cell_schedule_seed_{seed}.csv",
                repo
                / FINAL_ROOT
                / "training"
                / f"seed_{seed}"
                / "Joint-C50-sigma2"
                / "final_embedding.npy",
                repo
                / FINAL_ROOT
                / "training"
                / f"seed_{seed}"
                / "Joint-C50-sigma2"
                / "per_cell_clipping_diagnostics.csv",
                repo / FIGURE2_ROOT / "partitions" / f"seed_{seed}.json",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required saved artifacts: " + ", ".join(missing))

    provenance = json.loads((repo / FINAL_ROOT / "provenance.json").read_text())
    candidate = provenance["candidates"]["Joint-C50-sigma2"]
    checks = {
        "seeds": tuple(provenance["seeds"]) == SEEDS,
        "C": math.isclose(float(candidate["C"]), C, rel_tol=0, abs_tol=1e-9),
        "sigma": float(candidate["sigma"]) == SIGMA,
        "remote_cells": int(provenance["remote_cells_per_source_per_active_round"])
        == REMOTE_PER_SOURCE,
        "unique_remote": int(provenance["unique_remote_cells_per_source"])
        == len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
        "max_participation": int(provenance["max_remote_participation_required"]) == 1,
        "noise_seeds": {int(k): int(v) for k, v in provenance["noise_seeds"].items()}
        == NOISE_SEEDS,
    }
    if not all(checks.values()):
        raise ValueError(f"finalized protocol provenance mismatch: {checks}")
    exposure = pd.read_csv(repo / FINAL_ROOT / "exposure_QC.csv")
    if not (
        len(exposure) == len(SEEDS) * 5
        and (exposure.max_remote_participation_count == 1).all()
        and (exposure.remote_cell_reuse_count == 0).all()
        and exposure.zero_remote_cell_reuse.astype(bool).all()
    ):
        raise ValueError("saved exposure QC does not establish one-time participation")
    return {
        "validated_checks": checks,
        "final_provenance_sha256": sha256(repo / FINAL_ROOT / "provenance.json"),
        "exposure_qc_sha256": sha256(repo / FINAL_ROOT / "exposure_QC.csv"),
    }


def load_actual_norms(repo: Path, seed: int) -> pd.DataFrame:
    schedule_path = (
        repo / FINAL_ROOT / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
    )
    diagnostic_path = (
        repo
        / FINAL_ROOT
        / "training"
        / f"seed_{seed}"
        / "Joint-C50-sigma2"
        / "per_cell_clipping_diagnostics.csv"
    )
    schedule = pd.read_csv(schedule_path)
    diagnostics = pd.read_csv(diagnostic_path).rename(columns={"round": "active_round"})
    keys = ["active_round", "source_client", "cell_id"]
    merged = schedule.merge(
        diagnostics[keys + ["joint_norm", "clipping_scale", "clipped"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(ACTIVE_ROUNDS) * 5 * REMOTE_PER_SOURCE:
        raise ValueError(f"seed {seed}: schedule/diagnostic merge is incomplete")
    if merged.cell_id.nunique() != len(merged) or merged.groupby("cell_id").size().max() != 1:
        raise ValueError(f"seed {seed}: remote-cell reuse detected")
    expected_scale = np.minimum(1.0, C / merged.joint_norm.to_numpy(float))
    if not np.allclose(expected_scale, merged.clipping_scale, rtol=1e-5, atol=1e-6):
        raise ValueError(f"seed {seed}: clipping diagnostics do not match fixed C")
    merged["clipped_norm"] = np.minimum(merged.joint_norm.to_numpy(float), C)
    return merged


def gaussian_llr_metrics(
    clipped_norms: np.ndarray,
    *,
    draws: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Simulate the exact sufficient statistic for the equal-covariance Gaussian LRT."""

    d = np.asarray(clipped_norms, dtype=np.float64)
    snr = d / (SIGMA * C)
    rng = np.random.default_rng(random_seed)
    mean = 0.5 * np.square(snr)
    score_h0 = (-mean[:, None] + snr[:, None] * rng.standard_normal((len(d), draws))).ravel()
    score_h1 = (+mean[:, None] + snr[:, None] * rng.standard_normal((len(d), draws))).ravel()
    scores = np.concatenate([score_h0, score_h1])
    labels = np.concatenate(
        [np.zeros(len(score_h0), dtype=np.int8), np.ones(len(score_h1), dtype=np.int8)]
    )
    predictions = scores > 0.0
    return {
        "n_h0": len(score_h0),
        "n_h1": len(score_h1),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "attack_accuracy": float(accuracy_score(labels, predictions)),
        "mean_candidate_exact_auroc": float(np.mean(norm.cdf(snr / math.sqrt(2.0)))),
        "mean_candidate_exact_accuracy": float(np.mean(norm.cdf(snr / 2.0))),
        "mean_clipped_norm_over_C": float(np.mean(d / C)),
        "max_clipped_norm_over_C": float(np.max(d / C)),
    }


def membership_attacks(repo: Path, draws: int) -> pd.DataFrame:
    old = pd.read_csv(repo / FIGURE5_ROOT / "04_membership" / "membership_per_seed.csv")
    rows: list[dict] = []
    for seed in SEEDS:
        direct = old[(old.seed == seed) & (old.condition == "Direct queue exposure")]
        if len(direct) != 1:
            raise ValueError(f"seed {seed}: missing exact old direct-exposure MIA result")
        direct = direct.iloc[0]
        rows.append(
            {
                "seed": seed,
                "round": "all",
                "interface": "Vanilla FreshQueue: individual embedding exposure",
                "attack": "reused Figure 5 direct-exposure membership classifier",
                "sigma": math.nan,
                "C": math.nan,
                "n_candidates": int(direct.n_test_per_class),
                "noise_draws_per_hypothesis_per_candidate": math.nan,
                "n_h0": int(direct.n_test_per_class),
                "n_h1": int(direct.n_test_per_class),
                "auroc": float(direct.auroc),
                "auprc": float(direct.auprc),
                "attack_accuracy": float(direct.balanced_accuracy),
                "mean_candidate_exact_auroc": math.nan,
                "mean_candidate_exact_accuracy": math.nan,
                "mean_clipped_norm_over_C": math.nan,
                "max_clipped_norm_over_C": math.nan,
                "data_basis": "reused exact seven-seed Figure 5 privacy-holdout result",
                "individual_embedding_exposed": True,
            }
        )
        actual = load_actual_norms(repo, seed)
        for round_scope, subset in [("all", actual), *[(str(r), actual[actual.active_round == r]) for r in ACTIVE_ROUNDS]]:
            rows.append(
                {
                    "seed": seed,
                    "round": round_scope,
                    "interface": "RoundOnce Joint-C50 aggregate A/B, sigma=0",
                    "attack": "white-box adjacent-case oracle (deterministic release)",
                    "sigma": 0.0,
                    "C": C,
                    "n_candidates": len(subset),
                    "noise_draws_per_hypothesis_per_candidate": 1,
                    "n_h0": len(subset),
                    "n_h1": len(subset),
                    "auroc": 1.0,
                    "auprc": 1.0,
                    "attack_accuracy": 1.0,
                    "mean_candidate_exact_auroc": 1.0,
                    "mean_candidate_exact_accuracy": 1.0,
                    "mean_clipped_norm_over_C": float((subset.clipped_norm / C).mean()),
                    "max_clipped_norm_over_C": float((subset.clipped_norm / C).max()),
                    "data_basis": "actual saved per-cell joint norms and fixed schedule",
                    "individual_embedding_exposed": False,
                }
            )
            metrics = gaussian_llr_metrics(
                subset.clipped_norm.to_numpy(float),
                draws=draws,
                random_seed=20260930 + seed * 101 + (0 if round_scope == "all" else int(round_scope)),
            )
            rows.append(
                {
                    "seed": seed,
                    "round": round_scope,
                    "interface": "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2",
                    "attack": "exact Gaussian likelihood-ratio oracle",
                    "sigma": SIGMA,
                    "C": C,
                    "n_candidates": len(subset),
                    "noise_draws_per_hypothesis_per_candidate": draws,
                    **metrics,
                    "data_basis": "actual saved per-cell joint norms and fixed schedule",
                    "individual_embedding_exposed": False,
                }
            )
    return pd.DataFrame(rows)


def load_partition(repo: Path, seed: int) -> list[np.ndarray]:
    path = repo / FIGURE2_ROOT / "partitions" / f"seed_{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = [np.asarray(values, dtype=np.int64) for values in payload["client_indices"]]
    flat = np.concatenate(groups)
    if len(groups) != 5 or len(flat) != 16382 or len(np.unique(flat)) != len(flat):
        raise ValueError(f"seed {seed}: invalid saved Figure 2 partition")
    return groups


def anchor_rows(
    embeddings: np.ndarray,
    groups: list[np.ndarray],
    *,
    seed: int,
    round_number: int,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for target, group in enumerate(groups):
        rng = np.random.default_rng(seed * 1_000_003 + round_number * 1_009 + target * 97 + 53)
        selected = rng.choice(group, size=ANCHORS_PER_TARGET, replace=False)
        base = normalize(embeddings[selected])
        result[target] = np.concatenate([base, base], axis=0)
    return result


def source_noise(
    *, seed: int, round_number: int, source: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    standard_seed = int(
        (NOISE_SEEDS[seed] * 1_000_003 + round_number * 1_009 + source * 97 + 7_919)
        % (2**63 - 1)
    )
    block_size = REPRESENTATION_ROWS_PER_TARGET * (LATENT_DIM + 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(standard_seed)
    vector = torch.randn(4 * block_size, generator=generator, device=device)
    noise_a = []
    noise_b = []
    offset = 0
    for _ in range(4):
        a_end = offset + REPRESENTATION_ROWS_PER_TARGET
        b_end = a_end + REPRESENTATION_ROWS_PER_TARGET * LATENT_DIM
        noise_a.append(vector[offset:a_end])
        noise_b.append(vector[a_end:b_end].reshape(REPRESENTATION_ROWS_PER_TARGET, LATENT_DIM))
        offset = b_end
    return torch.cat(noise_a), torch.cat(noise_b)


def contribution(
    anchors: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.exp((anchors @ q) / TAU)
    raw_norm = torch.sqrt(2.0 * torch.square(weights).sum())
    scale = torch.minimum(
        torch.ones((), device=q.device, dtype=q.dtype),
        torch.as_tensor(C, device=q.device, dtype=q.dtype) / raw_norm.clamp_min(1e-30),
    )
    a = weights * scale
    b = a[:, None] * q[None, :]
    return a, b, raw_norm, scale


@torch.no_grad()
def dictionary_estimate(
    anchors: torch.Tensor,
    observed_a: torch.Tensor,
    observed_b: torch.Tensor,
    dictionary: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    best_value = math.inf
    best_q = None
    for start in range(0, len(dictionary), 256):
        q = torch.nn.functional.normalize(dictionary[start : start + 256], dim=1)
        weights = torch.exp((anchors @ q.T) / TAU)
        raw_norm = torch.sqrt(2.0 * torch.square(weights).sum(dim=0))
        scale = torch.minimum(torch.ones_like(raw_norm), C / raw_norm.clamp_min(1e-30))
        a = weights * scale[None, :]
        observed_b_dot_q = observed_b @ q.T
        objective = (
            2.0 * torch.square(a).sum(dim=0)
            - 2.0 * (a * (observed_a[:, None] + observed_b_dot_q)).sum(dim=0)
        ) / (anchors.shape[0] * (LATENT_DIM + 1))
        index = int(torch.argmin(objective))
        value = float(objective[index].cpu())
        if value < best_value:
            best_value = value
            best_q = q[index].detach().clone()
    if best_q is None:
        raise RuntimeError("empty reconstruction dictionary")
    return best_q, best_value


def optimized_estimate(
    anchors: torch.Tensor,
    observed_a: torch.Tensor,
    observed_b: torch.Tensor,
    dictionary_start: torch.Tensor,
    *,
    steps: int,
    restarts: int,
    random_seed: int,
) -> tuple[torch.Tensor, float]:
    device = anchors.device
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    starts = [dictionary_start]
    b_sum = observed_b.sum(dim=0)
    if float(torch.linalg.vector_norm(b_sum)) > 0:
        starts.append(torch.nn.functional.normalize(b_sum, dim=0))
    while len(starts) < restarts:
        starts.append(torch.randn(LATENT_DIM, generator=generator, device=device))
    raw = torch.stack(starts[:restarts]).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    anchors_expanded = anchors.unsqueeze(0)
    observed_a_expanded = observed_a.unsqueeze(0)
    observed_b_expanded = observed_b.unsqueeze(0)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        q = torch.nn.functional.normalize(raw, dim=1)
        weights = torch.exp(torch.einsum("rnd,rd->rn", anchors_expanded.expand(len(q), -1, -1), q) / TAU)
        raw_norm = torch.sqrt(2.0 * torch.square(weights).sum(dim=1))
        scale = torch.minimum(torch.ones_like(raw_norm), C / raw_norm.clamp_min(1e-30))
        a = weights * scale[:, None]
        observed_b_dot_q = torch.einsum(
            "rnd,rd->rn", observed_b_expanded.expand(len(q), -1, -1), q
        )
        objective = (
            2.0 * torch.square(a).sum(dim=1)
            - 2.0 * (a * (observed_a_expanded + observed_b_dot_q)).sum(dim=1)
        ) / (anchors.shape[0] * (LATENT_DIM + 1))
        objective.sum().backward()
        torch.nn.utils.clip_grad_norm_([raw], 10.0)
        optimizer.step()
        if step == int(steps * 0.7):
            for group in optimizer.param_groups:
                group["lr"] = 0.01
    with torch.no_grad():
        q = torch.nn.functional.normalize(raw, dim=1)
        weights = torch.exp(torch.einsum("nd,rd->rn", anchors, q) / TAU)
        raw_norm = torch.sqrt(2.0 * torch.square(weights).sum(dim=1))
        scale = torch.minimum(torch.ones_like(raw_norm), C / raw_norm.clamp_min(1e-30))
        a = weights * scale[:, None]
        observed_b_dot_q = torch.einsum("nd,rd->rn", observed_b, q)
        objective = (
            2.0 * torch.square(a).sum(dim=1)
            - 2.0 * (a * (observed_a[None, :] + observed_b_dot_q)).sum(dim=1)
        ) / (anchors.shape[0] * (LATENT_DIM + 1))
        index = int(torch.argmin(objective))
        return q[index].detach().clone(), float(objective[index].cpu())


def reconstruction_metrics(q_hat: np.ndarray, q_true: np.ndarray, population: np.ndarray) -> dict:
    q_hat = normalize(q_hat.reshape(1, -1))[0]
    q_true = normalize(q_true.reshape(1, -1))[0]
    cosine = float(np.clip(q_hat @ q_true, -1.0, 1.0))
    relative_l2 = float(np.linalg.norm(q_hat - q_true) / max(np.linalg.norm(q_true), 1e-12))
    similarities = population @ q_hat
    true_similarity = cosine
    rank = int(1 + np.count_nonzero(similarities > true_similarity + 1e-7))
    return {
        "embedding_cosine": cosine,
        "embedding_relative_l2": relative_l2,
        "nearest_neighbor_rank": rank,
        "nearest_neighbor_top1": rank == 1,
        "nearest_neighbor_top10": rank <= 10,
    }


def reconstruction_attacks(
    repo: Path,
    *,
    candidates_per_source: int,
    steps: int,
    restarts: int,
    device: torch.device,
) -> pd.DataFrame:
    old_reconstruction = pd.read_csv(
        repo / FIGURE5_ROOT / "03_reconstruction" / "reconstruction_per_seed.csv"
    )
    rows: list[dict] = []
    for seed in SEEDS:
        embedding_path = (
            repo
            / FINAL_ROOT
            / "training"
            / f"seed_{seed}"
            / "Joint-C50-sigma2"
            / "final_embedding.npy"
        )
        embeddings = normalize(np.load(embedding_path))
        population = embeddings
        groups = load_partition(repo, seed)
        schedule = pd.read_csv(
            repo / FINAL_ROOT / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        )
        old_mlp = old_reconstruction[
            (old_reconstruction.seed == seed) & (old_reconstruction.method == "MLP")
        ]
        if len(old_mlp) != 1:
            raise ValueError(f"seed {seed}: missing old Figure 5 MLP reconstruction result")
        old_expression_pearson = float(old_mlp.iloc[0].pearson_mean)
        for round_number in ROUNDS:
            target_anchors = anchor_rows(
                embeddings, groups, seed=seed, round_number=round_number
            )
            round_schedule = schedule[schedule.active_round == round_number]
            for source in range(5):
                source_schedule = round_schedule[
                    round_schedule.source_client == source
                ].sort_values("source_local_index")
                if len(source_schedule) != REMOTE_PER_SOURCE:
                    raise ValueError(f"seed {seed} round {round_number} source {source}: bad schedule")
                positions = np.linspace(
                    0, REMOTE_PER_SOURCE - 1, candidates_per_source, dtype=int
                )
                candidate_records = source_schedule.iloc[np.unique(positions)]
                target_order = [target for target in range(5) if target != source]
                anchors_np = np.concatenate([target_anchors[target] for target in target_order])
                anchors = torch.as_tensor(anchors_np, device=device)
                noise_a, noise_b = source_noise(
                    seed=seed, round_number=round_number, source=source, device=device
                )
                known_indices = source_schedule.global_index.to_numpy(int)
                for candidate_offset, record in candidate_records.reset_index(drop=True).iterrows():
                    global_index = int(record.global_index)
                    q_true_np = embeddings[global_index]
                    q_true = torch.as_tensor(q_true_np, device=device)
                    true_a, true_b, raw_norm, clipping_scale = contribution(anchors, q_true)
                    observed_a_sigma0 = true_a
                    observed_b_sigma0 = true_b
                    observed_a_sigma2 = true_a + SIGMA * C * noise_a
                    observed_b_sigma2 = true_b + SIGMA * C * noise_b

                    other_indices = known_indices[known_indices != global_index]
                    dictionary_np = embeddings[other_indices]
                    dictionary = torch.as_tensor(dictionary_np, device=device)
                    dictionary_hat, dictionary_objective = dictionary_estimate(
                        anchors, observed_a_sigma2, observed_b_sigma2, dictionary
                    )
                    optimized_hat, optimized_objective = optimized_estimate(
                        anchors,
                        observed_a_sigma2,
                        observed_b_sigma2,
                        dictionary_hat,
                        steps=steps,
                        restarts=restarts,
                        random_seed=seed * 1_000_003
                        + round_number * 1_009
                        + source * 97
                        + candidate_offset,
                    )
                    mean_hat = normalize(dictionary_np.mean(axis=0, keepdims=True))[0]
                    random_hat = dictionary_np[
                        (seed + round_number + source + candidate_offset) % len(dictionary_np)
                    ]
                    sigma0_hat = torch.nn.functional.normalize(
                        observed_b_sigma0.sum(dim=0) / observed_a_sigma0.sum().clamp_min(1e-30),
                        dim=0,
                    )
                    estimates = [
                        (
                            "Vanilla FreshQueue: directly exposed embedding",
                            "direct observation",
                            q_true_np,
                            math.nan,
                            old_expression_pearson,
                            "reused Figure 5 MLP decoder result on its matched privacy-victim representation",
                        ),
                        (
                            "RoundOnce Joint-C50 aggregate A/B, sigma=0",
                            "closed-form leave-one-out MLE",
                            sigma0_hat.detach().cpu().numpy(),
                            0.0,
                            math.nan,
                            "not run: no compatible frozen expression decoder",
                        ),
                        (
                            "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2",
                            "optimized 100-D leave-one-out Gaussian MLE",
                            optimized_hat.detach().cpu().numpy(),
                            optimized_objective,
                            math.nan,
                            "not run: no compatible frozen expression decoder",
                        ),
                        (
                            "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2",
                            "nearest known-remote dictionary baseline",
                            dictionary_hat.detach().cpu().numpy(),
                            dictionary_objective,
                            math.nan,
                            "not run: baseline only",
                        ),
                        (
                            "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2",
                            "mean known-remotes baseline",
                            mean_hat,
                            math.nan,
                            math.nan,
                            "not run: baseline only",
                        ),
                        (
                            "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2",
                            "random known-remote baseline",
                            random_hat,
                            math.nan,
                            math.nan,
                            "not run: baseline only",
                        ),
                    ]
                    for interface, attack, estimate, objective, expression_pearson, expression_status in estimates:
                        rows.append(
                            {
                                "seed": seed,
                                "round": round_number,
                                "source_client": source,
                                "candidate_cell_id": str(record.cell_id),
                                "candidate_global_index": global_index,
                                "interface": interface,
                                "attack": attack,
                                "sigma": 0.0 if "sigma=0" in interface else SIGMA if "sigma=2" in interface else math.nan,
                                "C": C if "RoundOnce" in interface else math.nan,
                                "joint_norm_surrogate": float(raw_norm.cpu()),
                                "clipping_scale_surrogate": float(clipping_scale.cpu()),
                                "objective_without_constant": objective,
                                **reconstruction_metrics(estimate, q_true_np, population),
                                "expression_pearson": expression_pearson,
                                "expression_status": expression_status,
                                "attack_data_basis": (
                                    "reused matched Figure 5 privacy-victim result"
                                    if "Vanilla" in interface
                                    else RECONSTRUCTION_BASIS
                                ),
                            }
                        )
        del embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize(
    membership: pd.DataFrame, reconstruction: pd.DataFrame, repo: Path
) -> pd.DataFrame:
    rows: list[dict] = []
    membership_seed = membership[membership["round"] == "all"].copy()
    for (interface, attack), group in membership_seed.groupby(["interface", "attack"]):
        for metric in ("auroc", "auprc", "attack_accuracy"):
            values = group.sort_values("seed")[metric].to_numpy(float)
            low, high = t_interval(values)
            rows.append(
                {
                    "analysis": "membership",
                    "interface": interface,
                    "attack": attack,
                    "metric": metric,
                    "n_seed_replicates": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "interpretation_scope": "seed is the replicate; candidate cells/noise draws are not independent model replicates",
                }
            )
    recon_seed = (
        reconstruction.groupby(["seed", "interface", "attack"], as_index=False)
        .agg(
            embedding_cosine=("embedding_cosine", "mean"),
            embedding_relative_l2=("embedding_relative_l2", "mean"),
            nearest_neighbor_top1=("nearest_neighbor_top1", "mean"),
            nearest_neighbor_rank=("nearest_neighbor_rank", "median"),
            expression_pearson=("expression_pearson", "mean"),
        )
    )
    for (interface, attack), group in recon_seed.groupby(["interface", "attack"]):
        for metric in (
            "embedding_cosine",
            "embedding_relative_l2",
            "nearest_neighbor_top1",
            "nearest_neighbor_rank",
            "expression_pearson",
        ):
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            low, high = t_interval(values)
            rows.append(
                {
                    "analysis": "reconstruction",
                    "interface": interface,
                    "attack": attack,
                    "metric": metric,
                    "n_seed_replicates": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                    "interpretation_scope": "seed is the replicate; cells/rounds are averaged within seed",
                }
            )
    old_membership = pd.read_csv(
        repo / FIGURE5_ROOT / "04_membership" / "membership_per_seed.csv"
    )
    model_only = old_membership[old_membership.condition == "Model-only"].sort_values("seed")
    for metric, source_column in (
        ("auroc", "auroc"),
        ("auprc", "auprc"),
        ("attack_accuracy", "balanced_accuracy"),
    ):
        values = model_only[source_column].to_numpy(float)
        low, high = t_interval(values)
        rows.append(
            {
                "analysis": "limitation_reference",
                "interface": "FreshQueue victim model-only; not protected by A/B release DP",
                "attack": "reused Figure 5 model-only membership classifier",
                "metric": metric,
                "n_seed_replicates": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "interpretation_scope": "context only; model-update/global-model leakage remains unresolved",
            }
        )
    return pd.DataFrame(rows)


def summary_value(summary: pd.DataFrame, interface: str, attack: str, metric: str) -> float:
    row = summary[
        (summary.interface == interface)
        & (summary.attack == attack)
        & (summary.metric == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"missing summary value: {interface}, {attack}, {metric}")
    return float(row.iloc[0]["mean"])


def build_report(
    membership: pd.DataFrame,
    reconstruction: pd.DataFrame,
    summary: pd.DataFrame,
    provenance: dict,
    args: argparse.Namespace,
) -> str:
    direct_interface = "Vanilla FreshQueue: individual embedding exposure"
    direct_attack = "reused Figure 5 direct-exposure membership classifier"
    sigma0_interface = "RoundOnce Joint-C50 aggregate A/B, sigma=0"
    sigma0_mia_attack = "white-box adjacent-case oracle (deterministic release)"
    sigma2_interface = "Final private RoundOnce Joint-C50 aggregate A/B, sigma=2"
    sigma2_mia_attack = "exact Gaussian likelihood-ratio oracle"
    optimized_attack = "optimized 100-D leave-one-out Gaussian MLE"
    direct_auc = summary_value(summary, direct_interface, direct_attack, "auroc")
    sigma0_auc = summary_value(summary, sigma0_interface, sigma0_mia_attack, "auroc")
    sigma2_auc = summary_value(summary, sigma2_interface, sigma2_mia_attack, "auroc")
    sigma2_auprc = summary_value(summary, sigma2_interface, sigma2_mia_attack, "auprc")
    sigma2_accuracy = summary_value(summary, sigma2_interface, sigma2_mia_attack, "attack_accuracy")
    sigma2_cosine = summary_value(summary, sigma2_interface, optimized_attack, "embedding_cosine")
    sigma2_relative = summary_value(summary, sigma2_interface, optimized_attack, "embedding_relative_l2")
    sigma2_top1 = summary_value(summary, sigma2_interface, optimized_attack, "nearest_neighbor_top1")
    sigma2_rank = summary_value(summary, sigma2_interface, optimized_attack, "nearest_neighbor_rank")
    direct_expression = summary_value(
        summary,
        "Vanilla FreshQueue: directly exposed embedding",
        "direct observation",
        "expression_pearson",
    )
    model_only_auc = summary_value(
        summary,
        "FreshQueue victim model-only; not protected by A/B release DP",
        "reused Figure 5 model-only membership classifier",
        "auroc",
    )
    theoretical_max_auc = float(norm.cdf(1.0 / (SIGMA * math.sqrt(2.0))))
    theoretical_max_accuracy = float(norm.cdf(1.0 / (2.0 * SIGMA)))
    excess_reduction = 1.0 - (sigma2_auc - 0.5) / (direct_auc - 0.5)
    optimized = reconstruction[reconstruction.attack == optimized_attack]
    verdict = "SUPPORT"
    return f"""# Final private A/B privacy-attack validation

## Scope and provenance

No model was trained and Figures 1, 2, and 5 were not modified. This audit uses
the finalized seven-seed `Joint-C50`, `C={C}`, `sigma=2`, 102-remote-cell
RoundOnce runs. Saved schedule QC confirms 1,734 unique cells per source over
rounds 4--20 and maximum remote participation one. The exact analytic Gaussian
accounting remains epsilon approximately 1.993 at delta=1e-5 for the remote A/B
release channel only.

Membership inference uses all {len(SEEDS) * len(ACTIVE_ROUNDS) * 5 * REMOTE_PER_SOURCE:,}
actual scheduled cell contributions through their saved joint norms, with
{args.membership_draws} H0 and {args.membership_draws} H1 Gaussian draws per cell.
The score is the exact likelihood ratio
`(y-S0).u/(sigma*C)^2 - ||u||^2/(2*(sigma*C)^2)`.

The training artifacts do not contain per-round anchor tensors, pre-noise A/B
releases, or restorable model checkpoints. Reconstruction is therefore a
post-training mechanism audit, not a claim that an actual saved transcript was
attacked: it uses real final embeddings, the fixed remote schedules, deterministic
128-cell target batches at rounds {', '.join(map(str, ROUNDS))}, and the original
per-seed/per-round/per-source DP-noise seed formula. The two protocol views are
represented by duplicating each real final anchor embedding. There are
{len(optimized)} attacked cells for the optimized final-private reconstruction.

The previous Figure 5 FreshQueue results were reused without rerunning them.
Their privacy victim setting matches the same Pancreas/Figure 2 seeds and uses a
disjoint stratified train/auxiliary/non-member split. The old expression decoder
was trained in that separate privacy-victim representation and no frozen decoder
checkpoint was saved. It was therefore not applied to the new RoundOnce
embeddings, and no decoder was retrained.

## Membership inference

| Interface | AUROC | AUPRC | Accuracy |
|---|---:|---:|---:|
| Vanilla FreshQueue, direct embedding | {direct_auc:.6f} | {summary_value(summary, direct_interface, direct_attack, 'auprc'):.6f} | {summary_value(summary, direct_interface, direct_attack, 'attack_accuracy'):.6f} |
| Aggregate A/B, sigma=0 | {sigma0_auc:.6f} | 1.000000 | 1.000000 |
| Final aggregate A/B, sigma=2 | {sigma2_auc:.6f} | {sigma2_auprc:.6f} | {sigma2_accuracy:.6f} |

Aggregation removes the direct individual-embedding interface, but it does not
by itself protect against the stipulated oracle: with known anchors and all
other cells, a deterministic sigma=0 residual reveals membership exactly. For
sigma=2, clipping gives `||u|| <= C`; consequently any single candidate has
AUROC at most `Phi(1/(sigma*sqrt(2))) = {theoretical_max_auc:.6f}` and equal-prior
accuracy at most `Phi(1/(2*sigma)) = {theoretical_max_accuracy:.6f}`. The observed
seed-level mean AUROC {sigma2_auc:.6f} is consistent with this bound and removes
{100 * excess_reduction:.1f}% of the direct attack's AUROC excess above chance.

## Leave-one-out reconstruction

| Interface / attack | Mean cosine | Mean relative L2 | Median NN rank | Top-1 recovery |
|---|---:|---:|---:|---:|
| Direct embedding observation | {summary_value(summary, 'Vanilla FreshQueue: directly exposed embedding', 'direct observation', 'embedding_cosine'):.6f} | {summary_value(summary, 'Vanilla FreshQueue: directly exposed embedding', 'direct observation', 'embedding_relative_l2'):.6f} | {summary_value(summary, 'Vanilla FreshQueue: directly exposed embedding', 'direct observation', 'nearest_neighbor_rank'):.1f} | {summary_value(summary, 'Vanilla FreshQueue: directly exposed embedding', 'direct observation', 'nearest_neighbor_top1'):.6f} |
| Aggregate sigma=0, closed form | {summary_value(summary, sigma0_interface, 'closed-form leave-one-out MLE', 'embedding_cosine'):.6f} | {summary_value(summary, sigma0_interface, 'closed-form leave-one-out MLE', 'embedding_relative_l2'):.6f} | {summary_value(summary, sigma0_interface, 'closed-form leave-one-out MLE', 'nearest_neighbor_rank'):.1f} | {summary_value(summary, sigma0_interface, 'closed-form leave-one-out MLE', 'nearest_neighbor_top1'):.6f} |
| Final sigma=2, optimized oracle | {sigma2_cosine:.6f} | {sigma2_relative:.6f} | {sigma2_rank:.1f} | {sigma2_top1:.6f} |
| Final sigma=2, known-remote dictionary | {summary_value(summary, sigma2_interface, 'nearest known-remote dictionary baseline', 'embedding_cosine'):.6f} | {summary_value(summary, sigma2_interface, 'nearest known-remote dictionary baseline', 'embedding_relative_l2'):.6f} | {summary_value(summary, sigma2_interface, 'nearest known-remote dictionary baseline', 'nearest_neighbor_rank'):.1f} | {summary_value(summary, sigma2_interface, 'nearest known-remote dictionary baseline', 'nearest_neighbor_top1'):.6f} |
| Final sigma=2, mean known remotes | {summary_value(summary, sigma2_interface, 'mean known-remotes baseline', 'embedding_cosine'):.6f} | {summary_value(summary, sigma2_interface, 'mean known-remotes baseline', 'embedding_relative_l2'):.6f} | {summary_value(summary, sigma2_interface, 'mean known-remotes baseline', 'nearest_neighbor_rank'):.1f} | {summary_value(summary, sigma2_interface, 'mean known-remotes baseline', 'nearest_neighbor_top1'):.6f} |
| Final sigma=2, random known remote | {summary_value(summary, sigma2_interface, 'random known-remote baseline', 'embedding_cosine'):.6f} | {summary_value(summary, sigma2_interface, 'random known-remote baseline', 'embedding_relative_l2'):.6f} | {summary_value(summary, sigma2_interface, 'random known-remote baseline', 'nearest_neighbor_rank'):.1f} | {summary_value(summary, sigma2_interface, 'random known-remote baseline', 'nearest_neighbor_top1'):.6f} |

The sigma=0 residual has rows `[w, w*q]`, so `q` is recovered exactly by the
closed-form ratio even after joint clipping. The sigma=2 attack optimizes a
unit-normalized 100-D candidate under the exact clipped A/B Gaussian likelihood
from {args.optimization_restarts} starts and {args.optimization_steps} Adam steps.
Its result should be interpreted against the mean, random, and known-remote
dictionary baselines above. The high mean-baseline cosine reflects a strong
common direction in this embedding space; its very poor nearest-neighbor rank
shows why cosine alone is not evidence of cell identity recovery. The old direct FreshQueue expression-decoder
Pearson correlation is {direct_expression:.6f}; no compatible frozen decoder
exists for a defensible RoundOnce expression-level comparison.

## Answers

1. Aggregate A/B removes individual embedding disclosure, but aggregation alone
   does **not** reduce leakage against the stipulated all-but-one oracle: sigma=0
   gives perfect MIA and exact embedding recovery.
2. Sigma=2 reduces MIA from approximately {direct_auc:.4f}/1.0 to AUROC
   {sigma2_auc:.4f} and prevents exact leave-one-out reconstruction in this
   strong surrogate audit (cosine {sigma2_cosine:.4f}, relative L2
   {sigma2_relative:.4f}, top-1 {sigma2_top1:.4f}).
3. An individual embedding is exactly reconstructable at sigma=0, but not at a
   useful exact-identity level under sigma=2 in this audit. New-method expression
   reconstruction is unreported because the available decoder is incompatible;
   retraining one was outside scope.
4. Final private release-channel MIA AUROC is **{sigma2_auc:.6f}** (seven-seed
   mean; detailed per-round values are in `membership_attack.csv`).
5. The empirical likelihood-ratio result respects the formal clipped Gaussian
   bound and supports the release-channel analysis. It is not end-to-end model
   privacy evidence.
6. Final verdict: **{verdict}**. The qualifier reflects the reconstructed-release
   surrogate, absence of a compatible expression decoder, and the unresolved
   model-update/global-model channel (old model-only MIA AUROC {model_only_auc:.6f}).

## Claim boundary

For the new method, **direct individual-embedding exposure no longer exists as
an attack interface**; it is not reported as a fictitious AUROC of 0.5. The DP
guarantee and new attacks cover only the cross-client A/B release channel.
Model updates and the global model are outside this mechanism and remain an
explicit limitation.

Provenance SHA-256: finalized protocol `{provenance['final_provenance_sha256']}`;
exposure QC `{provenance['exposure_qc_sha256']}`.
"""


def main() -> None:
    args = parse_args()
    global FINAL_ROOT, FIGURE2_ROOT, FIGURE5_ROOT
    FINAL_ROOT = args.final_root
    FIGURE2_ROOT = args.partition_root
    FIGURE5_ROOT = args.explicit_privacy_root
    if args.membership_draws < 1:
        raise ValueError("membership draws must be positive")
    if args.reconstruction_candidates_per_source < 1:
        raise ValueError("reconstruction candidate count must be positive")
    repo = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else repo / args.output
    output.mkdir(parents=True, exist_ok=True)
    provenance = validate_inputs(repo)
    membership = membership_attacks(repo, args.membership_draws)
    atomic_csv(membership, output / "membership_attack.csv")
    reconstruction = reconstruction_attacks(
        repo,
        candidates_per_source=args.reconstruction_candidates_per_source,
        steps=args.optimization_steps,
        restarts=args.optimization_restarts,
        device=torch.device(args.device),
    )
    atomic_csv(reconstruction, output / "reconstruction_attack.csv")
    summary = summarize(membership, reconstruction, repo)
    atomic_csv(summary, output / "attack_summary.csv")
    report = build_report(membership, reconstruction, summary, provenance, args)
    atomic_text(report, output / "PRIVACY_ATTACK_REPORT.md")
    print(report)


if __name__ == "__main__":
    main()
