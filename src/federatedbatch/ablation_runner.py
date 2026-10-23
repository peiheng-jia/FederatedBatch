"""Corrected queue, temperature, local-epoch, and partition ablations."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import REPO_ROOT, load_config
from .data import preprocess_dataset, read_dataset
from .evaluation import clear_evaluation_artifacts, evaluate_embedding
from .federated import train_federated
from .partition import partition_batches, partition_cells
from .provenance import build_metadata, write_json
from .runner import make_run_dir, save_resolved_config
from .statistics import summarize_metrics


def _partition_summary(adata, index_groups) -> list[dict]:
    summaries = []
    for client_index, indices in enumerate(index_groups):
        obs = adata.obs.iloc[indices]
        cell_ids = "\n".join(adata.obs_names[indices].astype(str))
        summaries.append(
            {
                "client": client_index,
                "n_cells": len(indices),
                "cell_ids_sha256": hashlib.sha256(cell_ids.encode()).hexdigest(),
                "batch_counts": {
                    str(key): int(value) for key, value in obs["batch"].value_counts().items()
                },
                "celltype_counts": {
                    str(key): int(value) for key, value in obs["celltype"].value_counts().items()
                },
            }
        )
    return summaries


def run_ablation(
    ablation_config: dict,
    *,
    verify_hash: bool = True,
    resume_dir: str | Path | None = None,
) -> Path:
    config = load_config(ablation_config["base_config"])
    if resume_dir is None:
        output_dir = make_run_dir(config, "ablation")
        save_resolved_config(config, output_dir / "base_config.yaml")
        write_json(output_dir / "ablation_grid.json", ablation_config)
        records = []
    else:
        output_dir = Path(resume_dir)
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        output_dir = output_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {output_dir}")
        with (output_dir / "ablation_grid.json").open(encoding="utf-8") as handle:
            saved_grid = json.load(handle)
        if saved_grid != ablation_config:
            raise ValueError("resume directory uses a different ablation grid")
        with (output_dir / "base_config.yaml").open(encoding="utf-8") as handle:
            saved_config = yaml.safe_load(handle)
        resolved_config = {key: value for key, value in config.items() if not key.startswith("_")}
        if saved_config != resolved_config:
            raise ValueError("resume directory uses a different base configuration")
        runs_path = output_dir / "ablation_runs.csv"
        records = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []

    adata, data_path = read_dataset(
        config["dataset"],
        config["data_dir"],
        verify_hash=verify_hash,
    )
    features = preprocess_dataset(adata, config)
    completed = {
        (str(record["experiment"]), str(record["parameter"]), int(record["seed"]))
        for record in records
    }

    conditions = []
    for value in ablation_config["queue_sizes"]:
        conditions.append(("queue_size", value))
    for value in ablation_config["temperatures"]:
        conditions.append(("temperature", value))
    for value in ablation_config["local_epochs"]:
        conditions.append(("local_epochs", value))
    for value in ablation_config["partition_schemes"]:
        conditions.append(("partition", value))
    for value in ablation_config.get("queue_stress", {}).get("queue_sizes", []):
        conditions.append(("queue_stress", value))

    for experiment, value in conditions:
        for seed in ablation_config["seeds"]:
            condition_key = (str(experiment), str(value), int(seed))
            if condition_key in completed:
                print(f"skip completed: {experiment}={value}, seed={seed}")
                continue
            run_config = copy.deepcopy(config)
            run_config["seed"] = seed
            if experiment == "queue_size":
                run_config["federated"]["queue"]["enabled"] = value > 0
                run_config["federated"]["queue"]["size"] = value
            elif experiment == "temperature":
                run_config["federated"]["temperature"] = value
            elif experiment == "local_epochs":
                run_config["federated"]["local_epochs"] = value
            elif experiment == "queue_stress":
                stress = ablation_config["queue_stress"]
                run_config["federated"]["batch_size"] = stress["batch_size"]
                run_config["federated"]["n_clients"] = stress["n_clients"]
                run_config["federated"]["queue"]["enabled"] = value > 0
                run_config["federated"]["queue"]["size"] = value

            client_indices = None
            if experiment in {"partition", "queue_stress"}:
                if experiment == "partition":
                    partition_config = ablation_config["partition"]
                    partition_scheme = str(value)
                else:
                    partition_config = ablation_config["queue_stress"]
                    partition_scheme = partition_config["partition_scheme"]
                run_config["federated"]["n_clients"] = partition_config["n_clients"]
                client_indices = partition_cells(
                    adata,
                    partition_config["n_clients"],
                    partition_scheme,
                    seed=seed,
                    label_skew_fraction=partition_config["label_skew_fraction"],
                    primary_clients_per_label=partition_config["primary_clients_per_label"],
                    quantity_skew_factor=partition_config["quantity_skew_factor"],
                )
                batches = None
                index_groups = client_indices
                partition_protocol = f"cell_{partition_scheme}"
            else:
                batches = partition_batches(
                    adata,
                    run_config["federated"]["n_clients"],
                    scheme="batch",
                    seed=seed,
                )
                index_groups = [
                    np.flatnonzero(np.asarray(adata.obs["batch"].isin(group)))
                    for group in batches
                ]
                partition_protocol = "non_overlapping_batch_ownership"
            condition_dir = output_dir / f"{experiment}_{value}" / f"seed_{seed}"
            result = train_federated(
                adata,
                features,
                batches,
                run_config,
                condition_dir,
                domain_key="batch",
                client_indices=client_indices,
            )
            embedding_key = f"X_{experiment}_{value}_{seed}"
            metrics = evaluate_embedding(
                adata,
                result.embeddings,
                run_config,
                embedding_key=embedding_key,
            )
            records.append(
                {
                    "experiment": experiment,
                    "parameter": value,
                    "seed": seed,
                    "n_clients": run_config["federated"]["n_clients"],
                    "queue_final": result.queue_sizes[-1],
                    "time_seconds": result.training_seconds,
                    **metrics,
                }
            )
            pd.DataFrame(records).to_csv(output_dir / "ablation_runs.csv", index=False)
            write_json(
                condition_dir / "training_summary.json",
                {
                    "partition_protocol": partition_protocol,
                    "clients": _partition_summary(adata, index_groups),
                    "queue_sizes": result.queue_sizes,
                    "skipped_state_keys": result.skipped_state_keys,
                    "training_seconds": result.training_seconds,
                },
            )
            clear_evaluation_artifacts(adata, embedding_key)
            del result, metrics
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(records)
    numeric = [
        "time_seconds",
        "ARI",
        "NMI",
        "ASW_batch",
        "ASW_celltype",
        "Graph_connectivity",
        "kNN_batch_mixing_scaled",
        "Label_transfer_accuracy",
        "Label_transfer_macro_F1",
        "Label_transfer_coverage",
    ]
    summary = raw.groupby(["experiment", "parameter"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "ablation_results.csv", index=False)
    summarize_metrics(raw, ["experiment", "parameter"]).to_csv(
        output_dir / "ablation_statistics.csv",
        index=False,
    )
    write_json(output_dir / "provenance.json", build_metadata(REPO_ROOT, data_path))
    return output_dir
