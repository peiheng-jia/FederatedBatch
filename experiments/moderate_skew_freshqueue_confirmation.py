"""Run the frozen independent confirmation of ordinary FreshQueue."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shlex
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from experiments.fresh_queue_regime import atomic_csv, evaluate_embedding
except ModuleNotFoundError:
    from fresh_queue_regime import atomic_csv, evaluate_embedding

from federatedbatch.config import REPO_ROOT, load_config, resolve_repo_path
from federatedbatch.data import preprocess_dataset, read_dataset
from federatedbatch.federated import predict_embeddings, train_federated
from federatedbatch.partition import partition_cells
from federatedbatch.provenance import build_metadata, git_revision, write_json
from federatedbatch.queue_regime import PAIR_METRICS, composition_statistics
from federatedbatch.runner import save_resolved_config

METHODS = ("NoQueue", "FreshQueue")
SEEDS = tuple(range(800, 810))
SETTING = "pancreas_moderate_label_skew_bs128"


def training_seeds(config: dict) -> tuple[int, ...]:
    seeds = tuple(config["confirmation"]["training_seeds"])
    if not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("training seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if any(seed < 0 or seed > 1000 for seed in seeds):
        raise ValueError("training seeds must be between 0 and 1000")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/moderate_skew_freshqueue_confirmation.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def validate_protocol(config: dict) -> None:
    fed = config["federated"]
    queue = fed["queue"]
    confirmation = config["confirmation"]
    observed_fed = (
        fed["n_clients"],
        fed["n_rounds"],
        fed["local_epochs"],
        fed["batch_size"],
        fed["learning_rate"],
        fed["temperature"],
        fed["hard_negative_beta"],
        fed["latent_dim"],
        fed["encoder_dims"],
        fed["dropout"],
    )
    expected_fed = (5, 20, 5, 128, 0.001, 0.07, 1.0, 100, [128], 0.1)
    if observed_fed != expected_fed:
        raise ValueError(f"frozen training protocol changed: {observed_fed}")
    expected_queue = {
        "enabled": True,
        "mode": "fresh_cross_client",
        "size": 1000,
        "samples_per_client": 128,
        "weight": 0.1,
        "temperature": 0.2,
        "mask_fraction": 0.05,
        "warmup_rounds": 3,
        "cross_client_only": True,
        "collection": "after_fedavg_with_aggregated_encoder",
    }
    if {key: queue[key] for key in expected_queue} != expected_queue:
        raise ValueError("ordinary FreshQueue protocol changed")
    if confirmation["condition"] != SETTING:
        raise ValueError("confirmation condition changed")
    training_seeds(config)
    if confirmation["methods"] != list(METHODS):
        raise ValueError("confirmation methods changed")
    if confirmation["final_round"] != 20:
        raise ValueError("primary endpoint must remain round 20")
    if confirmation["partition"] != {
        "scheme": "label_skew",
        "seed_source": "training_seed",
        "label_skew_fraction": 0.70,
        "primary_clients_per_label": 2,
    }:
        raise ValueError("moderate label-skew partition protocol changed")
    if confirmation["leiden_random_states"] != [0, 1, 2, 3, 4]:
        raise ValueError("Leiden evaluation seeds changed")
    if (
        confirmation["automatic_checkpoint_selection"]
        or confirmation["automatic_follow_up"]
        or confirmation["allow_seed_replacement"]
    ):
        raise ValueError("checkpoint selection, follow-up, and seed replacement are forbidden")


def method_config(config: dict, method: str, seed: int) -> dict:
    resolved = copy.deepcopy(config)
    resolved["seed"] = int(seed)
    resolved["federated"]["queue"]["enabled"] = method == "FreshQueue"
    resolved["analysis"] = {
        "leiden_random_states": list(
            resolved["confirmation"]["leiden_random_states"]
        ),
        "purity_k": int(resolved["confirmation"]["purity_k"]),
    }
    return resolved


def partition_for_seed(adata, config: dict, seed: int) -> list[np.ndarray]:
    partition = config["confirmation"]["partition"]
    return partition_cells(
        adata,
        int(config["federated"]["n_clients"]),
        partition["scheme"],
        seed=int(seed),
        label_skew_fraction=float(partition["label_skew_fraction"]),
        primary_clients_per_label=int(partition["primary_clients_per_label"]),
    )


def _partition_hash(groups: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for client, indices in enumerate(groups):
        digest.update(f"client={client}\n".encode())
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def freeze_preflight(
    adata, config: dict, output: Path, seeds: tuple[int, ...]
) -> dict[int, list[np.ndarray]]:
    partitions_dir = output / "partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    all_partitions: dict[int, list[np.ndarray]] = {}
    composition_rows = []
    descriptor_rows = []
    client_rows = []
    pairwise_rows = []
    for seed in seeds:
        groups = partition_for_seed(adata, config, seed)
        all_partitions[seed] = groups
        composition, descriptors = composition_statistics(labels, groups)
        cell_types = np.unique(labels)
        distributions = []
        for indices in groups:
            client_labels = labels[np.asarray(indices, dtype=np.int64)]
            counts = np.asarray(
                [(client_labels == cell_type).sum() for cell_type in cell_types],
                dtype=float,
            )
            distributions.append(counts / counts.sum())
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                shared = int(
                    np.sum(
                        (distributions[first] > 0) & (distributions[second] > 0)
                    )
                )
                pairwise_rows.append(
                    {
                        "seed": seed,
                        "client_a": first,
                        "client_b": second,
                        "cell_type_overlap": float(
                            np.minimum(
                                distributions[first], distributions[second]
                            ).sum()
                        ),
                        "jensen_shannon_divergence": float(
                            jensenshannon(
                                distributions[first], distributions[second], base=2.0
                            )
                            ** 2
                        ),
                        "shared_cell_types": shared,
                    }
                )
        client_types = []
        for row in composition:
            composition_rows.append({"seed": seed, **row})
            if int(row["cell_count"]) > 0:
                client_types.append((int(row["client"]), str(row["cell_type"])))
        sizes = [int(len(indices)) for indices in groups]
        for client, size in enumerate(sizes):
            observed_types = sum(first == client for first, _ in client_types)
            client_rows.append(
                {
                    "seed": seed,
                    "client": client,
                    "cell_count": size,
                    "observed_cell_types": observed_types,
                }
            )
        shared_types = sum(
            sum((labels[np.asarray(indices)] == cell_type).any() for indices in groups) > 1
            for cell_type in np.unique(labels)
        )
        descriptor_rows.append(
            {
                "seed": seed,
                "n_clients": len(groups),
                "total_cells": int(sum(sizes)),
                "client_cells_min": int(min(sizes)),
                "client_cells_max": int(max(sizes)),
                "client_cells_mean": float(np.mean(sizes)),
                "client_cells_sd": float(np.std(sizes, ddof=1)),
                "client_imbalance_max_min_ratio": float(max(sizes) / min(sizes)),
                "number_of_cell_types_shared_by_multiple_clients": int(shared_types),
                "partition_sha256": _partition_hash(groups),
                **descriptors,
            }
        )
        partition_payload = {
            "protocol": "label_skew",
            "seed": seed,
            "n_clients": 5,
            "label_skew_fraction": 0.70,
            "primary_clients_per_label": 2,
            "client_sizes": sizes,
            "partition_sha256": _partition_hash(groups),
            "client_indices": [np.asarray(indices, dtype=int).tolist() for indices in groups],
        }
        path = partitions_dir / f"seed_{seed}.json"
        serialized = json.dumps(partition_payload, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"frozen partition changed for seed {seed}")
        if not path.exists():
            path.write_text(serialized, encoding="utf-8")
    atomic_csv(pd.DataFrame(client_rows), output / "preflight_client_counts.csv")
    atomic_csv(pd.DataFrame(pairwise_rows), output / "preflight_pairwise_clients.csv")
    atomic_csv(
        pd.DataFrame(composition_rows), output / "preflight_client_celltype_composition.csv"
    )
    atomic_csv(pd.DataFrame(descriptor_rows), output / "preflight_partition_summary.csv")
    return all_partitions


def validate_centralized_reference(config: dict, data_sha256: str, output: Path) -> pd.DataFrame:
    reference = config["confirmation"]["centralized_reference"]
    path = Path(reference["path"])
    provenance_path = Path(reference["provenance_path"])
    if not path.exists() or not provenance_path.exists():
        raise FileNotFoundError("frozen centralized reference or provenance is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hash = reference["expected_data_sha256"]
    if data_sha256 != expected_hash or provenance["data"]["sha256"] != expected_hash:
        raise RuntimeError("centralized reference uses a different Pancreas dataset")
    frame = pd.read_csv(path)
    if len(frame) != int(reference["expected_n"]):
        raise RuntimeError("centralized reference seed count changed")
    required = set(PAIR_METRICS) | {"seed", "method"}
    if not required.issubset(frame.columns):
        raise RuntimeError("centralized reference evaluation columns are incomplete")
    if set(frame["method"]) != {"CentralizedCONCORD"}:
        raise RuntimeError("centralized reference contains unexpected methods")
    metadata = {
        "source": str(path),
        "seed_paired": False,
        "n": len(frame),
        "data_sha256": expected_hash,
        "interpretation": (
            "Frozen non-paired 30-seed centralized mean; used only for descriptive "
            "federation-gap and recovery calculations."
        ),
    }
    write_json(output / "centralized_reference_metadata.json", metadata)
    return frame


def trajectory_metrics(adata, embeddings, config: dict, identity: str) -> dict:
    key = f"X_modconfirm_{identity}"
    neighbors_key = f"neighbors_{key}"
    cluster_key = f"leiden_{key}"
    adata.obsm[key] = np.asarray(embeddings)
    sc.pp.neighbors(
        adata,
        n_neighbors=int(config["evaluation"]["n_neighbors"]),
        use_rep=key,
        random_state=0,
        key_added=neighbors_key,
    )
    sc.tl.leiden(
        adata,
        resolution=float(config["evaluation"]["leiden_resolution"]),
        random_state=int(config["confirmation"]["trajectory_leiden_seed"]),
        neighbors_key=neighbors_key,
        key_added=cluster_key,
    )
    labels = adata.obs["celltype"].astype(str).to_numpy()
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    result = {
        "ARI": float(adjusted_rand_score(labels, clusters)),
        "NMI": float(normalized_mutual_info_score(labels, clusters)),
    }
    metadata = adata.uns[neighbors_key]
    for graph_key in (metadata["connectivities_key"], metadata["distances_key"]):
        del adata.obsp[graph_key]
    del adata.uns[neighbors_key]
    del adata.obs[cluster_key]
    del adata.obsm[key]
    return result


def round_evaluator(adata, config: dict, rows: list[dict], identity: dict):
    def callback(round_number, model, _diagnostics):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            embedding = predict_embeddings(model, adata)
            rows.append(
                {
                    **identity,
                    "round": int(round_number),
                    **trajectory_metrics(
                        adata, embedding, config, f"{identity}_{round_number}"
                    ),
                }
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return callback


def read_rows(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def replace_rows(rows: list[dict], new_rows: list[dict], identity: dict) -> list[dict]:
    kept = [
        row
        for row in rows
        if not (
            int(row["seed"]) == int(identity["seed"])
            and row["method"] == identity["method"]
        )
    ]
    return [*kept, *new_rows]


def update_status(path: Path, identity: dict, status: str, error: str = "") -> None:
    rows = replace_rows(read_rows(path), [], identity)
    rows.append({**identity, "status": status, "error": error})
    atomic_csv(pd.DataFrame(rows).sort_values(["seed", "method"]), path)


def completed(rows: list[dict], identity: dict) -> bool:
    return any(
        int(row["seed"]) == int(identity["seed"])
        and row["method"] == identity["method"]
        for row in rows
    )


def embedding_path(output: Path, identity: dict) -> Path:
    return (
        output
        / "runs"
        / f"seed_{int(identity['seed'])}"
        / str(identity["method"])
        / "final_embedding.npy"
    )


def completed_with_artifacts(
    rows: list[dict],
    identity: dict,
    output: Path,
    save_embeddings: bool,
) -> bool:
    if not completed(rows, identity):
        return False
    return not save_embeddings or embedding_path(output, identity).is_file()


def save_embedding_atomic(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def paired_statistics(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else math.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
    margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else math.nan
    try:
        wilcoxon = float(stats.wilcoxon(values).pvalue) if n > 1 else math.nan
    except ValueError:
        wilcoxon = 1.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": float(np.median(values)) if n else math.nan,
        "ci95_low": mean - margin if n > 1 else math.nan,
        "ci95_high": mean + margin if n > 1 else math.nan,
        "positive_count": int(np.sum(values > 0)),
        "paired_t_pvalue": float(stats.ttest_1samp(values, 0).pvalue)
        if n > 1
        else math.nan,
        "wilcoxon_pvalue": wilcoxon,
    }


def build_seed_results(methods: pd.DataFrame, central: pd.DataFrame) -> pd.DataFrame:
    if methods.empty:
        return pd.DataFrame()
    central_means = {metric: float(central[metric].mean()) for metric in PAIR_METRICS}
    complete = methods.groupby("seed")["method"].nunique()
    complete_seeds = complete.loc[complete == len(METHODS)].index
    rows = []
    for seed in complete_seeds:
        indexed = methods.loc[methods["seed"] == seed].set_index("method")
        row = {"seed": int(seed)}
        for metric in PAIR_METRICS:
            noqueue = float(indexed.loc["NoQueue", metric])
            fresh = float(indexed.loc["FreshQueue", metric])
            short = metric.removesuffix("_mean")
            gap = central_means[metric] - noqueue
            recovery = fresh - noqueue
            row[f"NoQueue_{metric}"] = noqueue
            row[f"FreshQueue_{metric}"] = fresh
            row[f"Delta_{short}"] = recovery
            row[f"CentralizedReference_{metric}"] = central_means[metric]
            row[f"FederationGap_{short}"] = gap
            row[f"QueueRecovery_{short}"] = recovery
            row[f"RecoveryRatio_{short}"] = (
                recovery / gap
                if short in {"ARI", "NMI"} and gap > 0.02
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()


def build_summary(seed_results: pd.DataFrame, central: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in PAIR_METRICS:
        short = metric.removesuffix("_mean")
        values = seed_results[f"Delta_{short}"].to_numpy(float)
        rows.append(
            {
                "metric": short,
                "centralized_reference_n": len(central),
                "centralized_reference_mean": float(central[metric].mean()),
                "NoQueue_mean": float(seed_results[f"NoQueue_{metric}"].mean()),
                "FreshQueue_mean": float(seed_results[f"FreshQueue_{metric}"].mean()),
                "FederationGap_mean": float(
                    seed_results[f"FederationGap_{short}"].mean()
                ),
                "QueueRecovery_mean": float(
                    seed_results[f"QueueRecovery_{short}"].mean()
                ),
                "RecoveryRatio_descriptive": (
                    float(seed_results[f"QueueRecovery_{short}"].mean())
                    / float(seed_results[f"FederationGap_{short}"].mean())
                    if short in {"ARI", "NMI"}
                    and float(seed_results[f"FederationGap_{short}"].mean()) > 0.02
                    else math.nan
                ),
                **paired_statistics(values),
            }
        )
    return pd.DataFrame(rows)


def write_figures(seed_results: pd.DataFrame, trajectory: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not seed_results.empty:
        for metric in ("ARI", "NMI"):
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            for index, row in seed_results.reset_index(drop=True).iterrows():
                values = [row[f"NoQueue_{metric}_mean"], row[f"FreshQueue_{metric}_mean"]]
                axis.plot([0, 1], values, marker="o", alpha=0.72, label=str(int(row["seed"])))
            axis.set(
                xticks=[0, 1],
                xticklabels=["NoQueue", "FreshQueue"],
                ylabel=f"Final-round {metric}",
            )
            axis.grid(axis="y", alpha=0.2)
            axis.legend(title="Training seed", bbox_to_anchor=(1.02, 1), frameon=False)
            figure.tight_layout()
            figure.savefig(figures / f"paired_{metric}.png", dpi=180)
            plt.close(figure)
    if not trajectory.empty:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for axis, metric in zip(axes, ("ARI", "NMI")):
            for method in METHODS:
                subset = trajectory.loc[trajectory["method"] == method]
                grouped = subset.groupby("round")[metric]
                means = grouped.mean()
                axis.plot(means.index, means.values, marker="o", label=method)
            axis.set(xlabel="Communication round", ylabel=metric)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(figures / "round_trajectory.png", dpi=180)
        plt.close(figure)


def write_report(
    seed_results: pd.DataFrame,
    summary: pd.DataFrame,
    descriptors: pd.DataFrame,
    output: Path,
    seeds: tuple[int, ...],
    classification_label: str,
) -> None:
    complete = len(seed_results) == len(seeds)
    required_positive = math.ceil(0.7 * len(seeds))
    moderate_positive = math.ceil(0.6 * len(seeds))
    lines = [
        "# Moderate-skew ordinary FreshQueue "
        + classification_label.replace("_", " "),
        "",
        "Frozen protocol: Pancreas, five clients, label_skew_fraction=0.70, "
        "primary_clients_per_label=2, partition seed equals training seed, BS128, "
        "five local epochs, 20 communication rounds, and ordinary FreshQueue only.",
        "",
        f"Complete paired seed blocks: {len(seed_results)}/{len(seeds)}.",
        "",
        "The frozen partitions have mean pairwise overlap "
        f"{descriptors['mean_pairwise_composition_overlap'].mean():.4f} and mean "
        "pairwise Jensen-Shannon divergence "
        f"{descriptors['mean_pairwise_jensen_shannon_divergence'].mean():.4f}. "
        "Client sizes are intentionally inherited from the historical generator and "
        "are not forced to be equal.",
        "",
        "The centralized reference is a frozen, non-paired 30-seed reference on the "
        "same global Pancreas data and evaluation protocol. Gap and recovery ratios "
        "are descriptive; the primary test is paired FreshQueue minus NoQueue.",
        "",
        "## Paired results",
        "",
        "```text",
        summary.to_string(index=False, float_format=lambda value: f"{value:.5f}")
        if len(summary)
        else "No complete seed blocks yet.",
        "```",
        "",
        "## Conclusion",
        "",
    ]
    if classification_label == "posthoc_positive_seed_subset":
        lines[4:4] = [
            "This cohort was selected post hoc because both ARI and NMI improved in a "
            "previous run. It must not be interpreted as independent or confirmatory evidence.",
            "",
        ]
    if not complete:
        lines.append(
            f"Final interpretation is deferred until all {len(seeds)} frozen seeds complete."
        )
    else:
        indexed = summary.set_index("metric")
        ari = indexed.loc["ARI"]
        nmi = indexed.loc["NMI"]
        auxiliary = indexed.loc[
            [
                "ASW_cell",
                "ASW_batch",
                "graph_connectivity",
                "iLISI_proxy",
                "knn_celltype_purity_k15",
            ]
        ]
        systematic_harm = auxiliary.loc[
            (auxiliary["mean"] < 0)
            & ((len(seeds) - auxiliary["positive_count"]) >= required_positive)
        ]
        strong = (
            ari["mean"] > 0
            and ari["positive_count"] >= required_positive
            and ari["ci95_low"] > 0
            and nmi["mean"] >= 0
            and systematic_harm.empty
        )
        moderate = (
            ari["mean"] > 0
            and ari["positive_count"] >= moderate_positive
            and ari["positive_count"] <= required_positive
            and not strong
            and abs(ari["mean"] - 0.02) <= 0.02
        )
        classification = (
            "Strong confirmation"
            if strong
            else "Moderate evidence"
            if moderate
            else "Inconclusive positive trend"
            if ari["mean"] > 0 and ari["positive_count"] >= moderate_positive
            else "Failure to confirm"
        )
        lines.append(
            f"{classification}: mean delta ARI={ari['mean']:+.4f}, "
            f"95% CI [{ari['ci95_low']:+.4f}, {ari['ci95_high']:+.4f}], "
            f"{int(ari['positive_count'])}/{len(seeds)} positive seeds. Mean delta "
            f"NMI={nmi['mean']:+.4f}."
        )
        lines.append(
            "Other biological and batch metrics are reported without selecting or "
            "redefining the primary endpoint. No follow-up or queue modification was launched."
        )
        if systematic_harm.empty:
            lines.append(
                "No auxiliary metric met the prespecified systematic-harm rule "
                "(negative mean with at least 70% negative paired seeds)."
            )
        else:
            lines.append(
                "The prespecified systematic-harm rule was met for: "
                + ", ".join(systematic_harm.index.astype(str))
                + "."
            )
    (output / "moderate_skew_freshqueue_confirmation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def summarize(
    output: Path,
    central: pd.DataFrame,
    seeds: tuple[int, ...],
    classification_label: str,
) -> pd.DataFrame:
    methods_path = output / "method_metrics.csv"
    trajectory_path = output / "round_trajectory.csv"
    descriptor_path = output / "preflight_partition_summary.csv"
    methods = pd.read_csv(methods_path) if methods_path.exists() else pd.DataFrame()
    trajectory = (
        pd.read_csv(trajectory_path) if trajectory_path.exists() else pd.DataFrame()
    )
    descriptors = pd.read_csv(descriptor_path)
    seed_results = build_seed_results(methods, central)
    summary = build_summary(seed_results, central) if len(seed_results) else pd.DataFrame()
    atomic_csv(
        seed_results,
        output / "moderate_skew_freshqueue_confirmation_seed_results.csv",
    )
    atomic_csv(
        summary,
        output / "moderate_skew_freshqueue_confirmation_summary.csv",
    )
    write_figures(seed_results, trajectory, output)
    write_report(
        seed_results, summary, descriptors, output, seeds, classification_label
    )
    return seed_results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_protocol(config)
    seeds = training_seeds(config)
    classification_label = str(config["confirmation"]["classification"])
    output = resolve_repo_path(args.output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output / "config.yaml")
    adata, data_path = read_dataset(
        config["dataset"], config["data_dir"], verify_hash=not args.no_verify_hash
    )
    provenance = build_metadata(REPO_ROOT, data_path)
    write_json(output / "provenance.json", provenance)
    partitions = freeze_preflight(adata, config, output, seeds)
    central = validate_centralized_reference(
        config, provenance["data"]["sha256"], output
    )
    if args.preflight_only:
        print(f"Preflight: {output}")
        return

    features_path = output / "features.txt"
    if features_path.exists():
        features = features_path.read_text(encoding="utf-8").splitlines()
        if "log1p" not in adata.uns:
            from federatedbatch.data import normalize

            normalize(
                adata,
                config["preprocessing"]["normalize_target_sum"],
                config["preprocessing"]["log1p"],
            )
    else:
        features = preprocess_dataset(adata, config)
        features_path.write_text("\n".join(features) + "\n", encoding="utf-8")
    if args.summarize_only:
        summarize(output, central, seeds, classification_label)
        return

    manifest_path = output / "run_manifest.json"
    manifest = {
        "git_commit": git_revision(REPO_ROOT),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "pid": os.getpid(),
        "status": "started",
        "classification": classification_label,
        "expected_runs": len(seeds) * len(METHODS),
        "expected_seed_blocks": len(seeds),
        "primary_endpoint": "final_round_ARI",
        "automatic_follow_up": False,
    }
    write_json(manifest_path, manifest)
    paths = {
        "method": output / "method_metrics.csv",
        "repeats": output / "leiden_repeats.csv",
        "trajectory": output / "round_trajectory.csv",
        "queue": output / "training_queue_diagnostics.csv",
        "status": output / "run_status.csv",
    }
    rows = {key: read_rows(path) for key, path in paths.items() if key != "status"}
    save_embeddings = bool(config["evaluation"].get("save_embeddings", False))
    for seed in seeds:
        groups = partitions[seed]
        for method in METHODS:
            identity = {"seed": seed, "method": method}
            if completed_with_artifacts(
                rows["method"], identity, output, save_embeddings
            ):
                update_status(paths["status"], identity, "completed")
                continue
            update_status(paths["status"], identity, "running")
            print(f"\n=== seed={seed} method={method} ===", flush=True)
            configured = method_config(config, method, seed)
            run_dir = output / "runs" / f"seed_{seed}" / method
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_rows = []
            try:
                result = train_federated(
                    adata,
                    features,
                    None,
                    configured,
                    run_dir,
                    domain_key="batch",
                    client_indices=groups,
                    round_callback=round_evaluator(
                        adata, configured, trajectory_rows, identity
                    ),
                )
                if save_embeddings:
                    save_embedding_atomic(
                        embedding_path(output, identity), result.embeddings
                    )
                repeats, metrics = evaluate_embedding(
                    adata,
                    result.embeddings,
                    configured,
                    f"moderate_confirm_{seed}_{method}",
                )
            except Exception as exc:
                update_status(paths["status"], identity, "failed", repr(exc))
                manifest.update(
                    {"status": "failed", "failed_run": identity, "error": repr(exc)}
                )
                write_json(manifest_path, manifest)
                raise
            method_row = {
                **identity,
                "formal_evaluation_round": 20,
                "training_seconds": float(result.training_seconds),
                **metrics,
            }
            rows["method"] = replace_rows(rows["method"], [method_row], identity)
            rows["repeats"] = replace_rows(
                rows["repeats"],
                [{**identity, **row} for row in repeats],
                identity,
            )
            rows["trajectory"] = replace_rows(
                rows["trajectory"], trajectory_rows, identity
            )
            queue_rows = []
            for diagnostic in result.queue_diagnostics:
                queue_rows.append(
                    {
                        **identity,
                        **{
                            key: json.dumps(value)
                            if isinstance(value, (dict, list))
                            else value
                            for key, value in diagnostic.items()
                        },
                    }
                )
            rows["queue"] = replace_rows(rows["queue"], queue_rows, identity)
            for key, values in rows.items():
                atomic_csv(pd.DataFrame(values), paths[key])
            update_status(paths["status"], identity, "completed")
            summarize(output, central, seeds, classification_label)
            del result
            gc.collect()

    seed_results = summarize(output, central, seeds, classification_label)
    manifest.update(
        {
            "status": "completed",
            "completed_runs": len(rows["method"]),
            "completed_seed_blocks": len(seed_results),
            "automatic_follow_up_launched": False,
        }
    )
    write_json(manifest_path, manifest)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
