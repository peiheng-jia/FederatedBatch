from __future__ import annotations

from pathlib import Path
import copy
from typing import Iterable

import pandas as pd

from .baselines import run_centralized_zero_shot, run_pca_zero_shot
from .config import REPO_ROOT
from .data import normalize, read_dataset, select_features
from .evaluation import evaluate_embedding
from .federated import predict_embeddings, save_checkpoint, train_federated
from .partition import partition_batches
from .provenance import build_metadata, write_json
from .runner import resolve_run_dir, save_resolved_config


LOBO_METHOD_KEYS = {
    "federated": "FederatedBatch",
    "federated_no_queue": "FederatedBatch_no_queue",
    "centralized": "Centralized_CONCORD",
    "pca": "PCA",
}


def run_lobo(
    config: dict,
    *,
    verify_hash: bool = True,
    held_out_batches: list[str] | None = None,
    output_dir: str | Path | None = None,
    methods: Iterable[str] = ("federated",),
    resume: bool = False,
) -> Path:
    methods = list(methods)
    unknown = sorted(set(methods) - LOBO_METHOD_KEYS.keys())
    if unknown:
        raise ValueError(f"unknown LOBO methods: {', '.join(unknown)}")
    output_dir = resolve_run_dir(config, "lobo", output_dir, resume=resume)
    save_resolved_config(config, output_dir / "config.yaml")
    source, data_path = read_dataset(
        config["dataset"],
        config["data_dir"],
        verify_hash=verify_hash,
    )
    batches = list(source.obs["batch"].unique())
    if held_out_batches:
        unknown = sorted(set(held_out_batches) - set(batches))
        if unknown:
            raise ValueError(f"unknown held-out batches: {', '.join(unknown)}")
        batches = held_out_batches

    summary_path = output_dir / "lobo_summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if resume and summary_path.exists() else []
    for held_out in batches:
        completed_methods = {
            row["Method"]
            for row in rows
            if str(row["held_out_batch"]) == str(held_out)
        }
        pending_methods = [
            method for method in methods if LOBO_METHOD_KEYS[method] not in completed_methods
        ]
        if not pending_methods:
            print(f"skip completed held-out batch: {held_out}")
            continue
        print(f"\n=== held out: {held_out} ===")
        train = source[source.obs["batch"] != held_out].copy()
        test = source[source.obs["batch"] == held_out].copy()
        prep = config["preprocessing"]
        features = select_features(
            train,
            prep["n_hvg"],
            prep["hvg_flavor"],
        )
        normalize(train, prep["normalize_target_sum"], prep["log1p"])
        normalize(test, prep["normalize_target_sum"], prep["log1p"])
        missing = sorted(set(features) - set(test.var_names))
        if missing:
            raise ValueError(f"held-out batch {held_out} is missing {len(missing)} features")

        batch_dir = output_dir / str(held_out)
        batch_dir.mkdir(exist_ok=resume)
        (batch_dir / "features.txt").write_text(
            "\n".join(features) + "\n",
            encoding="utf-8",
        )
        client_batches = partition_batches(
            train,
            config["federated"]["n_clients"],
            scheme="batch",
            seed=config["seed"],
        )
        for method in pending_methods:
            print(f"--- {LOBO_METHOD_KEYS[method]} ---")
            if method in {"federated", "federated_no_queue"}:
                method_config = copy.deepcopy(config)
                training_name = "training"
                checkpoint_name = "global_model.pt"
                if method == "federated_no_queue":
                    method_config["federated"]["queue"]["enabled"] = False
                    method_config["federated"]["queue"]["size"] = 0
                    training_name = "training_no_queue"
                    checkpoint_name = "global_model_no_queue.pt"
                training_dir = batch_dir / training_name
                result = train_federated(
                    train,
                    features,
                    client_batches,
                    method_config,
                    training_dir,
                    domain_key=None,
                )
                write_json(
                    training_dir / "training_summary.json",
                    {
                        "client_batches": result.client_batches,
                        "client_cells": result.client_cells,
                        "queue_sizes": result.queue_sizes,
                        "skipped_state_keys": result.skipped_state_keys,
                        "training_seconds": result.training_seconds,
                    },
                )
                embeddings = predict_embeddings(result.model, test)
                elapsed = result.training_seconds
                save_checkpoint(
                    batch_dir / checkpoint_name,
                    result,
                    features,
                    method_config,
                    domain_key=None,
                )
            elif method == "centralized":
                embeddings, elapsed = run_centralized_zero_shot(
                    train,
                    test,
                    features,
                    config,
                    batch_dir / "centralized",
                )
            elif method == "pca":
                embeddings, elapsed = run_pca_zero_shot(train, test, features, config)
            else:
                raise AssertionError(f"unhandled LOBO method: {method}")

            embedding_key = f"X_lobo_{method}"
            metrics = evaluate_embedding(
                test,
                embeddings,
                config,
                embedding_key=embedding_key,
            )
            row = {
                "held_out_batch": held_out,
                "Method": LOBO_METHOD_KEYS[method],
                "n_train_cells": train.n_obs,
                "n_test_cells": test.n_obs,
                "training_time_seconds": elapsed,
                "communication_rounds_inference": 0,
                **metrics,
            }
            rows.append(row)
            write_json(batch_dir / f"metrics_{method}.json", row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)

        if config["evaluation"]["save_embeddings"]:
            test.write_h5ad(batch_dir / "held_out_with_embeddings.h5ad", compression="gzip")

    write_json(output_dir / "provenance.json", build_metadata(REPO_ROOT, data_path))
    return output_dir
