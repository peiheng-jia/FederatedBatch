from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .baselines import run_centralized, run_harmony, run_pca
from .config import REPO_ROOT, resolve_repo_path
from .data import preprocess_dataset, read_dataset
from .evaluation import evaluate_embedding
from .federated import save_checkpoint, train_federated
from .partition import partition_batches
from .provenance import build_metadata, write_json

METHOD_KEYS = {
    "federated": "FederatedBatch",
    "federated_no_queue": "FederatedBatch_no_queue",
    "centralized": "Centralized_CONCORD",
    "pca": "PCA",
    "harmony": "Harmony",
}


def make_run_dir(config: dict, experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_repo_path(config["output_dir"])
    path = output_root / f"{experiment}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_run_dir(
    config: dict,
    experiment: str,
    output_dir: str | Path | None,
    *,
    resume: bool = False,
) -> Path:
    if output_dir is None:
        return make_run_dir(config, experiment)
    path = Path(output_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=resume)
    return path


def save_resolved_config(config: dict, path: Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            saved = yaml.safe_load(handle)
        if saved != serializable:
            raise ValueError(f"saved configuration does not match the requested run: {path}")
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def run_methods(
    adata,
    features: list[str],
    config: dict,
    output_dir: Path,
    methods: Iterable[str],
    *,
    partition_scheme: str = "batch",
) -> list[dict]:
    methods = list(methods)
    unknown = sorted(set(methods) - METHOD_KEYS.keys())
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    client_batches = partition_batches(
        adata,
        config["federated"]["n_clients"],
        scheme=partition_scheme,
        seed=config["seed"],
    )
    rows = []
    for method in methods:
        print(f"\n=== {METHOD_KEYS[method]} ===")
        if method in {"federated", "federated_no_queue"}:
            method_config = copy.deepcopy(config)
            method_dir = "federatedbatch"
            if method == "federated_no_queue":
                method_config["federated"]["queue"]["enabled"] = False
                method_config["federated"]["queue"]["size"] = 0
                method_dir = "federatedbatch_no_queue"
            result = train_federated(
                adata,
                features,
                client_batches,
                method_config,
                output_dir / method_dir,
                domain_key="batch",
            )
            embeddings = result.embeddings
            elapsed = result.training_seconds
            save_checkpoint(
                output_dir / method_dir / "global_model.pt",
                result,
                features,
                method_config,
                domain_key="batch",
            )
            write_json(
                output_dir / method_dir / "training_summary.json",
                {
                    "client_batches": result.client_batches,
                    "client_cells": result.client_cells,
                    "queue_sizes": result.queue_sizes,
                    "skipped_state_keys": result.skipped_state_keys,
                    "training_seconds": result.training_seconds,
                },
            )
        elif method == "centralized":
            embeddings, elapsed = run_centralized(
                adata,
                features,
                config,
                output_dir / "centralized",
            )
        elif method == "pca":
            embeddings, elapsed = run_pca(adata, features, config)
        elif method == "harmony":
            embeddings, elapsed = run_harmony(adata, features, config)
        else:
            raise AssertionError(f"unhandled method: {method}")

        embedding_key = f"X_{method}"
        metrics = evaluate_embedding(
            adata,
            embeddings,
            config,
            embedding_key=embedding_key,
        )
        rows.append(
            {
                "Method": METHOD_KEYS[method],
                "n_clients": (
                    config["federated"]["n_clients"]
                    if method in {"federated", "federated_no_queue"}
                    else 1
                ),
                "training_time_seconds": elapsed,
                **metrics,
            }
        )
        print(json.dumps(rows[-1], indent=2))
    return rows


def run_main(
    config: dict,
    methods: Iterable[str],
    *,
    verify_hash: bool = True,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    output_dir = resolve_run_dir(
        config,
        f"main_{config['dataset']}",
        output_dir,
        resume=resume,
    )
    save_resolved_config(config, output_dir / "config.yaml")
    adata, data_path = read_dataset(
        config["dataset"],
        config["data_dir"],
        verify_hash=verify_hash,
    )
    features = preprocess_dataset(adata, config)
    (output_dir / "features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    results_path = output_dir / "results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if resume and results_path.exists() else []
    completed_methods = {row["Method"] for row in rows}
    for method in methods:
        if METHOD_KEYS[method] in completed_methods:
            print(f"skip completed method: {METHOD_KEYS[method]}")
            continue
        rows.extend(run_methods(adata, features, config, output_dir, [method]))
        pd.DataFrame(rows).to_csv(results_path, index=False)
    write_json(output_dir / "provenance.json", build_metadata(REPO_ROOT, data_path))
    if config["evaluation"]["save_embeddings"]:
        adata.write_h5ad(output_dir / "adata_with_embeddings.h5ad", compression="gzip")
    return output_dir


def run_multi_dataset(
    config: dict,
    methods: Iterable[str],
    datasets: Iterable[str],
    *,
    verify_hash: bool = True,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    output_dir = resolve_run_dir(config, "multi_dataset", output_dir, resume=resume)
    save_resolved_config(config, output_dir / "config.yaml")
    summary_path = output_dir / "multi_dataset_results.csv"
    all_rows = (
        pd.read_csv(summary_path).to_dict("records")
        if resume and summary_path.exists()
        else []
    )
    provenance_path = output_dir / "provenance.json"
    provenance = {}
    if resume and provenance_path.exists():
        with provenance_path.open(encoding="utf-8") as handle:
            provenance = json.load(handle).get("datasets", {})
    for dataset in datasets:
        dataset_config = copy.deepcopy(config)
        dataset_config["dataset"] = dataset
        expected_methods = {METHOD_KEYS[method] for method in methods}
        completed_methods = {
            row["Method"] for row in all_rows if str(row.get("dataset")) == dataset
        }
        pending_methods = [
            method for method in methods if METHOD_KEYS[method] not in completed_methods
        ]
        if not pending_methods:
            print(f"skip completed dataset: {dataset}")
            continue
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=resume)
        adata, data_path = read_dataset(
            dataset,
            dataset_config["data_dir"],
            verify_hash=verify_hash,
        )
        features = preprocess_dataset(adata, dataset_config)
        (dataset_dir / "features.txt").write_text(
            "\n".join(features) + "\n",
            encoding="utf-8",
        )
        dataset_rows = [row for row in all_rows if str(row.get("dataset")) == dataset]
        for method in pending_methods:
            new_rows = run_methods(adata, features, dataset_config, dataset_dir, [method])
            for row in new_rows:
                row.update(
                    {
                        "dataset": dataset,
                        "n_cells": adata.n_obs,
                        "n_batches": adata.obs["batch"].nunique(),
                        "n_celltypes": adata.obs["celltype"].nunique(),
                    }
                )
            dataset_rows.extend(new_rows)
            all_rows.extend(new_rows)
            pd.DataFrame(dataset_rows).to_csv(dataset_dir / "results.csv", index=False)
            pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        if {row["Method"] for row in dataset_rows} != expected_methods:
            raise RuntimeError(f"incomplete method set for {dataset}")
        if dataset_config["evaluation"]["save_embeddings"]:
            adata.write_h5ad(dataset_dir / "adata_with_embeddings.h5ad", compression="gzip")
        provenance[dataset] = build_metadata(REPO_ROOT, data_path)["data"]

    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    metadata = build_metadata(REPO_ROOT)
    metadata["datasets"] = provenance
    write_json(provenance_path, metadata)
    return output_dir
