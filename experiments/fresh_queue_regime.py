#!/usr/bin/env python3
"""Analyze the prespecified regimes in which fixed FreshQueue may be useful."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from federatedbatch.config import REPO_ROOT, load_config, resolve_repo_path
from federatedbatch.data import preprocess_dataset, read_dataset
from federatedbatch.evaluation import (
    graph_connectivity,
    knn_inverse_simpson,
    scib_silhouette_scores,
)
from federatedbatch.federated import train_federated
from federatedbatch.partition import partition_batches, partition_cells
from federatedbatch.provenance import build_metadata, git_revision, write_json
from federatedbatch.queue_regime import (
    RoundMechanismRecorder,
    candidate_settings,
    composition_statistics,
    paired_setting_results,
    regime_associations,
    setting_summary,
)
from federatedbatch.runner import save_resolved_config

METHODS = ("NoQueue", "FreshQueue")
SETTING_NAMES = (
    "pancreas_original_bs128",
    "pancreas_original_bs32",
    "pancreas_moderate_label_skew_bs128",
    "pancreas_strong_label_skew_bs128",
    "immune_original_bs128",
    "lung_original_bs128",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fresh_queue_regime.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--methods", choices=METHODS, nargs="+")
    parser.add_argument("--settings", choices=SETTING_NAMES, nargs="+")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def validate_protocol(config: dict) -> None:
    fed = config["federated"]
    queue = fed["queue"]
    observed_queue = {
        "mode": queue["mode"],
        "size": queue["size"],
        "samples_per_client": queue["samples_per_client"],
        "weight": queue["weight"],
        "temperature": queue["temperature"],
        "mask_fraction": queue["mask_fraction"],
        "warmup_rounds": queue["warmup_rounds"],
        "cross_client_only": queue["cross_client_only"],
        "collection": queue["collection"],
    }
    expected_queue = {
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
    if observed_queue != expected_queue:
        raise ValueError(f"fixed FreshQueue protocol changed: {observed_queue}")
    if fed["n_rounds"] != 8 or fed["local_epochs"] != 5:
        raise ValueError("regime analysis is fixed to 8 rounds and 5 local epochs")
    analysis = config["analysis"]
    if analysis["seeds"] != [42, 43, 44] or analysis["methods"] != list(METHODS):
        raise ValueError("paired seed or method protocol changed")
    if [setting["name"] for setting in analysis["settings"]] != list(SETTING_NAMES):
        raise ValueError("the six prespecified settings changed")
    expected_settings = {
        "pancreas_original_bs128": ("pancreas", "batch", 3, 128, None, None),
        "pancreas_original_bs32": ("pancreas", "batch", 3, 32, None, None),
        "pancreas_moderate_label_skew_bs128": (
            "pancreas",
            "label_skew",
            5,
            128,
            0.70,
            2,
        ),
        "pancreas_strong_label_skew_bs128": (
            "pancreas",
            "label_skew",
            5,
            128,
            0.85,
            1,
        ),
        "immune_original_bs128": ("immune", "batch", 3, 128, None, None),
        "lung_original_bs128": ("lung", "batch", 3, 128, None, None),
    }
    for setting in analysis["settings"]:
        observed = (
            setting["dataset"],
            setting["partition"],
            setting["n_clients"],
            setting["batch_size"],
            setting.get("label_skew_fraction"),
            setting.get("primary_clients_per_label"),
        )
        if observed != expected_settings[setting["name"]]:
            raise ValueError(f"prespecified setting changed: {setting['name']}={observed}")
    criteria = analysis["candidate_criteria"]
    if criteria != {
        "paired_seeds": 3,
        "positive_ari_seeds": 3,
        "minimum_mean_delta_ari": 0.02,
        "minimum_mean_delta_nmi": 0.0,
        "minimum_mean_delta_asw_cell": 0.0,
        "minimum_mean_delta_knn_purity": 0.0,
    }:
        raise ValueError("confirmation-candidate criteria changed")


def make_partition(adata, setting: dict, seed: int):
    n_clients = int(setting["n_clients"])
    if setting["partition"] == "batch":
        batches = partition_batches(adata, n_clients, seed=seed)
        indices = [
            np.flatnonzero(np.asarray(adata.obs["batch"].isin(group)))
            for group in batches
        ]
        return batches, indices
    if setting["partition"] == "label_skew":
        indices = partition_cells(
            adata,
            n_clients,
            "label_skew",
            seed=seed,
            label_skew_fraction=float(setting["label_skew_fraction"]),
            primary_clients_per_label=int(setting["primary_clients_per_label"]),
        )
        return None, indices
    raise ValueError(f"unknown partition: {setting['partition']}")


def fixed_reference_ids(
    adata,
    client_indices: list[np.ndarray],
    *,
    setting_name: str,
    seed: int,
    cells_per_client: int,
) -> list[np.ndarray]:
    digest = hashlib.sha256(setting_name.encode()).digest()
    setting_seed = int.from_bytes(digest[:4], byteorder="little")
    rng = np.random.default_rng(setting_seed + int(seed) * 1_000_003)
    references = []
    for indices in client_indices:
        count = min(cells_per_client, len(indices))
        selected = rng.choice(np.asarray(indices), size=count, replace=False)
        references.append(np.asarray(adata.obs_names[selected].astype(str)))
    return references


def exact_knn_purity(embeddings: np.ndarray, labels, k: int) -> float:
    labels = np.asarray(labels).astype(str)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    raw = model.fit(embeddings).kneighbors(embeddings, return_distance=False)
    neighbors = np.empty((len(embeddings), k), dtype=np.int64)
    for index, candidates in enumerate(raw):
        selected = candidates[candidates != index][:k]
        if len(selected) != k:
            raise RuntimeError(f"could not identify {k} non-self neighbors")
        neighbors[index] = selected
    return float(np.mean(labels[neighbors] == labels[:, None]))


def evaluate_embedding(adata, embeddings, config: dict, identity: str):
    key = f"X_regime_{identity}"
    neighbors_key = f"neighbors_{key}"
    adata.obsm[key] = np.asarray(embeddings)
    sc.pp.neighbors(
        adata,
        use_rep=key,
        n_neighbors=int(config["evaluation"]["n_neighbors"]),
        random_state=0,
        key_added=neighbors_key,
    )
    labels = np.asarray(adata.obs["celltype"])
    repeats = []
    for leiden_seed in config["analysis"]["leiden_random_states"]:
        cluster_key = f"leiden_{key}_{leiden_seed}"
        sc.tl.leiden(
            adata,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            resolution=1.0,
            random_state=int(leiden_seed),
        )
        clusters = np.asarray(adata.obs[cluster_key])
        repeats.append(
            {
                "leiden_random_state": int(leiden_seed),
                "ARI": float(adjusted_rand_score(labels, clusters)),
                "NMI": float(normalized_mutual_info_score(labels, clusters)),
                "cluster_number": int(len(np.unique(clusters))),
            }
        )
        del adata.obs[cluster_key]
    connectivities = adata.obsp[f"{neighbors_key}_connectivities"]
    silhouette = scib_silhouette_scores(
        embeddings, adata.obs["celltype"], adata.obs["batch"]
    )
    ilisi = knn_inverse_simpson(connectivities, adata.obs["batch"])
    n_batches = adata.obs["batch"].nunique()
    metrics = {
        "ARI_mean": float(np.mean([row["ARI"] for row in repeats])),
        "ARI_sd": float(np.std([row["ARI"] for row in repeats], ddof=1)),
        "NMI_mean": float(np.mean([row["NMI"] for row in repeats])),
        "NMI_sd": float(np.std([row["NMI"] for row in repeats], ddof=1)),
        "ASW_cell": silhouette["ASW_celltype"],
        "ASW_batch": silhouette["ASW_batch"],
        "graph_connectivity": graph_connectivity(
            connectivities, adata.obs["celltype"]
        ),
        "iLISI_proxy": ilisi,
        "iLISI_proxy_scaled": (ilisi - 1) / (n_batches - 1),
        "knn_celltype_purity_k15": exact_knn_purity(
            embeddings,
            adata.obs["celltype"],
            int(config["analysis"]["purity_k"]),
        ),
    }
    metadata = adata.uns[neighbors_key]
    for graph_key in (metadata["connectivities_key"], metadata["distances_key"]):
        del adata.obsp[graph_key]
    del adata.uns[neighbors_key]
    del adata.obsm[key]
    return repeats, metrics


def jsonify_lists(row: dict) -> dict:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def write_analysis(
    method_rows: list[dict],
    descriptor_rows: list[dict],
    round_rows: list[dict],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = paired_setting_results(
        pd.DataFrame(method_rows),
        pd.DataFrame(descriptor_rows),
        pd.DataFrame(round_rows),
    )
    summary = setting_summary(paired)
    candidates = candidate_settings(summary)
    atomic_csv(paired, output / "setting_seed_results.csv")
    atomic_csv(summary, output / "setting_summary.csv")
    atomic_csv(candidates, output / "confirmation_candidates.csv")
    atomic_csv(
        regime_associations(paired, summary), output / "regime_associations.csv"
    )
    return paired, summary, candidates


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_protocol(config)
    settings = list(config["analysis"]["settings"])
    if args.settings:
        settings = [setting for setting in settings if setting["name"] in args.settings]
    seeds = args.seeds or list(config["analysis"]["seeds"])
    methods = args.methods or list(config["analysis"]["methods"])
    if args.smoke_test:
        settings = [
            setting
            for setting in settings
            if setting["name"] == "pancreas_moderate_label_skew_bs128"
        ]
        if not settings:
            raise ValueError("smoke test requires pancreas_moderate_label_skew_bs128")
        seeds = [42]
        methods = ["FreshQueue"]
        config["federated"]["n_rounds"] = 4
        config["federated"]["local_epochs"] = 1

    output = resolve_repo_path(args.output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=args.resume)
    save_resolved_config(config, output / "config.yaml")
    manifest = {
        "git_commit": git_revision(REPO_ROOT),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "pid": os.getpid(),
        "config": str(Path(config["_config_path"])),
        "status": "started",
        "smoke_test": args.smoke_test,
    }
    write_json(output / "run_manifest.json", manifest)
    write_json(
        output / "metric_definitions.json",
        {
            "representation_drift": (
                "Mean cosine distance for the same fixed reference cells under adjacent "
                "post-FedAvg global encoders."
            ),
            "queue_false_negative_mass": (
                "Post-mask softmax mass from same-cell-type entries in the actual preceding-"
                "round queue, evaluated on fixed reference anchors and averaged over five "
                "deterministic balanced samples; labels are accessed only after training."
            ),
            "queue_false_negative_mass_pre_mask": (
                "The corresponding same-cell-type queue softmax mass before the fixed top-5% "
                "similarity mask."
            ),
            "mean_pairwise_composition_overlap": (
                "Mean client-pair histogram intersection sum_c min(p_i(c), p_j(c))."
            ),
            "mean_pairwise_jensen_shannon_divergence": (
                "Mean base-2 Jensen-Shannon divergence across client cell-type distributions."
            ),
            "iLISI_proxy": (
                "Weighted inverse Simpson index on the shared 15-neighbor graph; not standard "
                "scIB graph iLISI."
            ),
            "knn_celltype_purity_k15": (
                "Mean same-cell-type fraction among 15 exact Euclidean neighbors."
            ),
        },
    )
    method_path = output / "method_metrics.csv"
    repeat_path = output / "leiden_repeats.csv"
    composition_path = output / "client_celltype_composition.csv"
    descriptor_path = output / "composition_descriptors.csv"
    round_path = output / "round_mechanism_diagnostics.csv"
    queue_path = output / "training_queue_diagnostics.csv"
    method_rows = read_records(method_path)
    repeat_rows = read_records(repeat_path)
    composition_rows = read_records(composition_path)
    descriptor_rows = read_records(descriptor_path)
    round_rows = read_records(round_path)
    queue_rows = read_records(queue_path)

    for dataset in dict.fromkeys(setting["dataset"] for setting in settings):
        adata, data_path = read_dataset(
            dataset, config["data_dir"], verify_hash=not args.no_verify_hash
        )
        dataset_config = copy.deepcopy(config)
        dataset_config["dataset"] = dataset
        features = preprocess_dataset(adata, dataset_config)
        (output / f"features_{dataset}.txt").write_text(
            "\n".join(features) + "\n", encoding="utf-8"
        )
        write_json(
            output / f"provenance_{dataset}.json",
            build_metadata(REPO_ROOT, data_path),
        )
        for setting in [item for item in settings if item["dataset"] == dataset]:
            for seed in seeds:
                client_batches, client_indices = make_partition(adata, setting, int(seed))
                composition_rows = [
                    row
                    for row in composition_rows
                    if not (
                        row["setting"] == setting["name"]
                        and int(row["seed"]) == int(seed)
                    )
                ]
                descriptor_rows = [
                    row
                    for row in descriptor_rows
                    if not (
                        row["setting"] == setting["name"]
                        and int(row["seed"]) == int(seed)
                    )
                ]
                compositions, descriptors = composition_statistics(
                    np.asarray(adata.obs["celltype"]), client_indices
                )
                composition_rows.extend(
                    {
                        "setting": setting["name"],
                        "dataset": dataset,
                        "seed": int(seed),
                        **row,
                    }
                    for row in compositions
                )
                descriptor_rows.append(
                    {
                        "setting": setting["name"],
                        "dataset": dataset,
                        "seed": int(seed),
                        "partition": setting["partition"],
                        "n_clients": int(setting["n_clients"]),
                        "batch_size": int(setting["batch_size"]),
                        **descriptors,
                    }
                )
                atomic_csv(pd.DataFrame(composition_rows), composition_path)
                atomic_csv(pd.DataFrame(descriptor_rows), descriptor_path)
                references = fixed_reference_ids(
                    adata,
                    client_indices,
                    setting_name=setting["name"],
                    seed=int(seed),
                    cells_per_client=int(
                        config["analysis"]["reference_cells_per_client"]
                    ),
                )
                write_json(
                    output / f"partition_{setting['name']}_seed_{seed}.json",
                    {
                        "setting": setting,
                        "seed": int(seed),
                        "client_batches": client_batches,
                        "client_sizes": [int(len(indices)) for indices in client_indices],
                        "reference_cell_ids_sha256": [
                            hashlib.sha256("\n".join(ids).encode()).hexdigest()
                            for ids in references
                        ],
                    },
                )
                for method in methods:
                    complete = any(
                        row["setting"] == setting["name"]
                        and int(row["seed"]) == int(seed)
                        and row["method"] == method
                        for row in method_rows
                    )
                    if complete:
                        print(
                            f"skip setting={setting['name']}, seed={seed}, method={method}"
                        )
                        continue
                    method_rows = [
                        row
                        for row in method_rows
                        if not (
                            row["setting"] == setting["name"]
                            and int(row["seed"]) == int(seed)
                            and row["method"] == method
                        )
                    ]
                    round_rows = [
                        row
                        for row in round_rows
                        if not (
                            row["setting"] == setting["name"]
                            and int(row["seed"]) == int(seed)
                            and row["method"] == method
                        )
                    ]
                    queue_rows = [
                        row
                        for row in queue_rows
                        if not (
                            row["setting"] == setting["name"]
                            and int(row["seed"]) == int(seed)
                            and row["method"] == method
                        )
                    ]
                    method_config = copy.deepcopy(dataset_config)
                    method_config["seed"] = int(seed)
                    method_config["federated"]["n_clients"] = int(
                        setting["n_clients"]
                    )
                    method_config["federated"]["batch_size"] = int(
                        setting["batch_size"]
                    )
                    method_config["federated"]["queue"]["enabled"] = (
                        method == "FreshQueue"
                    )
                    method_dir = (
                        output / setting["name"] / f"seed_{seed}" / method
                    )
                    method_dir.mkdir(parents=True, exist_ok=args.resume)
                    recorder = RoundMechanismRecorder(
                        references, method=method, training_seed=int(seed)
                    )
                    print(
                        f"\n=== setting={setting['name']}, seed={seed}, method={method} ==="
                    )
                    result = train_federated(
                        adata,
                        features,
                        client_batches,
                        method_config,
                        method_dir,
                        domain_key="batch",
                        client_indices=(None if client_batches is not None else client_indices),
                        diagnostic_callback=recorder,
                    )
                    finalized = recorder.finalize(
                        adata,
                        n_rounds=int(method_config["federated"]["n_rounds"]),
                        warmup_rounds=int(
                            method_config["federated"]["queue"]["warmup_rounds"]
                        ),
                        batch_size=int(method_config["federated"]["batch_size"]),
                        temperature=float(
                            method_config["federated"]["queue"]["temperature"]
                        ),
                        mask_fraction=float(
                            method_config["federated"]["queue"]["mask_fraction"]
                        ),
                        sampling_repeats=int(
                            config["analysis"]["false_negative_sampling_repeats"]
                        ),
                    )
                    round_rows.extend(
                        {
                            "setting": setting["name"],
                            "dataset": dataset,
                            "seed": int(seed),
                            **row,
                        }
                        for row in finalized
                    )
                    atomic_csv(pd.DataFrame(round_rows), round_path)
                    for item in result.queue_diagnostics:
                        queue_rows.append(
                            jsonify_lists(
                                {
                                    "setting": setting["name"],
                                    "dataset": dataset,
                                    "seed": int(seed),
                                    "method": method,
                                    **item,
                                }
                            )
                        )
                    atomic_csv(pd.DataFrame(queue_rows), queue_path)
                    identity = f"{setting['name']}_{seed}_{method}"
                    repeats, metrics = evaluate_embedding(
                        adata, result.embeddings, method_config, identity
                    )
                    repeat_rows = [
                        row
                        for row in repeat_rows
                        if not (
                            row["setting"] == setting["name"]
                            and int(row["seed"]) == int(seed)
                            and row["method"] == method
                        )
                    ]
                    repeat_rows.extend(
                        {
                            "setting": setting["name"],
                            "dataset": dataset,
                            "seed": int(seed),
                            "method": method,
                            **row,
                        }
                        for row in repeats
                    )
                    method_rows.append(
                        {
                            "setting": setting["name"],
                            "dataset": dataset,
                            "seed": int(seed),
                            "partition": setting["partition"],
                            "n_clients": int(setting["n_clients"]),
                            "batch_size": int(setting["batch_size"]),
                            "method": method,
                            "training_seconds": result.training_seconds,
                            **metrics,
                        }
                    )
                    atomic_csv(pd.DataFrame(repeat_rows), repeat_path)
                    atomic_csv(pd.DataFrame(method_rows), method_path)
                    write_json(
                        method_dir / "training_summary.json",
                        {
                            "setting": setting,
                            "seed": int(seed),
                            "method": method,
                            "client_cells": result.client_cells,
                            "queue_sizes": result.queue_sizes,
                            "training_seconds": result.training_seconds,
                            "saved_model_checkpoint": False,
                        },
                    )
                    write_analysis(method_rows, descriptor_rows, round_rows, output)
        del adata
        gc.collect()

    paired, summary, candidates = write_analysis(
        method_rows, descriptor_rows, round_rows, output
    )
    selected_candidates = (
        candidates.loc[candidates["confirmation_candidate"], "setting"].tolist()
        if len(candidates)
        else []
    )
    manifest.update(
        {
            "status": "completed",
            "completed_setting_seed_pairs": int(len(paired)),
            "confirmation_candidates": selected_candidates,
            "automatic_follow_up_launched": False,
        }
    )
    write_json(output / "run_manifest.json", manifest)
    print(f"\nResults: {output}")
    print(f"Confirmation candidates: {selected_candidates}")


if __name__ == "__main__":
    main()
