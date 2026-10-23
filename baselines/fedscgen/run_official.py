#!/usr/bin/env python3
"""Run the pinned official FedscGen VAE under the manuscript protocol."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

try:
    from .official_adapter import (
        align_parts,
        assign_sites,
        balanced_subsample,
        evaluate,
        latent_embeddings,
        load_adapter_config,
        make_run_dir,
        make_test_client,
        official_two_stage_correction,
        preprocess,
        provenance,
        read_dataset,
        save_embedding_adata,
        seeded_site_groups,
        train_official,
        write_json,
    )
except ImportError:  # Direct execution: python baselines/fedscgen/run_official.py
    from official_adapter import (
        align_parts,
        assign_sites,
        balanced_subsample,
        evaluate,
        latent_embeddings,
        load_adapter_config,
        make_run_dir,
        make_test_client,
        official_two_stage_correction,
        preprocess,
        provenance,
        read_dataset,
        save_embedding_adata,
        seeded_site_groups,
        train_official,
        write_json,
    )


def apply_overrides(config, args):
    config = copy.deepcopy(config)
    for argument, key in (
        (args.n_rounds, "n_rounds"),
        (args.local_epochs, "local_epochs"),
    ):
        if argument is not None:
            config["training"][key] = argument
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def prepare_dataset(name, config, args, workflow):
    source, data_path = read_dataset(name, config, verify_hash=not args.no_verify_hash)
    if args.max_cells:
        source = balanced_subsample(source, args.max_cells, config["seed"])
    prepared, features = preprocess(
        source,
        config["preprocessing"][workflow],
        n_hvg=args.n_hvg,
    )
    return prepared, data_path, features


def save_common(directory: Path, config, features, mapping, data_path):
    (directory / "features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    write_json(directory / "site_mapping.json", mapping)
    write_json(directory / "provenance.json", provenance(config, data_path, features))


def run_multi(args) -> Path:
    config = apply_overrides(load_adapter_config(args.config), args)
    output = make_run_dir(config, "multi_dataset", args.output_dir, resume=args.resume)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    summary_path = output / "fedscgen_multi_dataset_results.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if args.resume and summary_path.exists() else []
    for name in args.datasets:
        if any(str(row["dataset"]) == name for row in rows):
            print(f"skip completed FedscGen dataset: {name}")
            continue
        print(f"\n=== Official FedscGen: {name} ===")
        directory = output / name
        directory.mkdir(exist_ok=args.resume)
        prepared, data_path, features = prepare_dataset(name, config, args, "multi_dataset")
        groups = seeded_site_groups(
            prepared,
            config["training"]["n_clients"],
            config["seed"],
        )
        mapping = assign_sites(prepared, groups)
        global_model, clients, training_seconds = train_official(prepared, config, directory)
        del global_model
        parts, correction_seconds = official_two_stage_correction(clients, config)
        corrected = align_parts(parts, prepared.obs_names)
        embeddings = corrected.obsm["latent_corrected"]
        metrics = evaluate(prepared, embeddings, config)
        row = {
            "dataset": name,
            "Method": "FedscGen",
            "n_cells": prepared.n_obs,
            "n_batches": prepared.obs["batch"].nunique(),
            "n_clients": config["training"]["n_clients"],
            "training_time_seconds": training_seconds,
            "correction_time_seconds": correction_seconds,
            "correction_communication_rounds": 2,
            **metrics,
        }
        rows.append(row)
        write_json(directory / "metrics.json", row)
        save_common(directory, config, features, mapping, data_path)
        if config["evaluation"]["save_embeddings"]:
            save_embedding_adata(prepared, embeddings, directory / "adata_with_embeddings.h5ad")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
    return output


def run_lobo(args) -> Path:
    config = apply_overrides(load_adapter_config(args.config), args)
    output = make_run_dir(config, "lobo_pancreas", args.output_dir, resume=args.resume)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    source, data_path = read_dataset("pancreas", config, verify_hash=not args.no_verify_hash)
    if args.max_cells:
        source = balanced_subsample(source, args.max_cells, config["seed"])
    held_out_batches = args.held_out_batches or list(source.obs["batch"].unique())

    summary_path = output / "fedscgen_lobo_summary.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if args.resume and summary_path.exists() else []
    for held_out in held_out_batches:
        completed_methods = {
            row["Method"]
            for row in rows
            if str(row["held_out_batch"]) == str(held_out)
        }
        expected_methods = {"FedscGen_zero_shot", "FedscGen_two_stage"}
        if completed_methods == expected_methods:
            print(f"skip completed FedscGen held-out batch: {held_out}")
            continue
        print(f"\n=== Official FedscGen LOBO: {held_out} ===")
        directory = output / str(held_out)
        directory.mkdir(exist_ok=args.resume)
        train_raw = source[source.obs["batch"] != held_out].copy()
        test_raw = source[source.obs["batch"] == held_out].copy()
        train, features = preprocess(
            train_raw,
            config["preprocessing"]["lobo"],
            n_hvg=args.n_hvg,
        )
        # Match the native LOBO path: normalize held-out cells over the full
        # gene space, then retain the HVGs selected from the training split.
        test = test_raw.copy()
        target_sum = config["preprocessing"]["lobo"]["normalize_target_sum"]
        if target_sum is None:
            import scanpy as sc

            sc.pp.normalize_total(test)
        else:
            import scanpy as sc

            sc.pp.normalize_total(test, target_sum=target_sum)
        if config["preprocessing"]["lobo"]["log1p"]:
            sc.pp.log1p(test)
        test = test[:, features].copy()
        test.raw = None

        groups = seeded_site_groups(train, config["training"]["n_clients"], config["seed"])
        mapping = assign_sites(train, groups)
        test.obs["fedscgen_site"] = "held_out"
        global_model, clients, training_seconds = train_official(train, config, directory)
        test_client = make_test_client(test, global_model, config, directory)
        zero_shot_embeddings = latent_embeddings(test_client, test)
        zero_shot_metrics = evaluate(test, zero_shot_embeddings, config)
        zero_shot_row = {
            "held_out_batch": held_out,
            "Method": "FedscGen_zero_shot",
            "deployment_mode": "zero_shot_encoding",
            "n_train_cells": train.n_obs,
            "n_test_cells": test.n_obs,
            "training_time_seconds": training_seconds,
            "correction_time_seconds": 0.0,
            "communication_rounds_inference": 0,
            **zero_shot_metrics,
        }
        if "FedscGen_zero_shot" not in completed_methods:
            rows.append(zero_shot_row)
            write_json(directory / "metrics_zero_shot.json", zero_shot_row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            if config["evaluation"]["save_embeddings"]:
                save_embedding_adata(
                    test,
                    zero_shot_embeddings,
                    directory / "held_out_zero_shot_embedding.h5ad",
                    embedding_key="X_fedscgen_zero_shot",
                )

        if "FedscGen_two_stage" not in completed_methods:
            parts, correction_seconds = official_two_stage_correction(
                [*clients, test_client],
                config,
            )
            corrected_test = align_parts([parts[-1]], test.obs_names)
            embeddings = corrected_test.obsm["latent_corrected"]
            metrics = evaluate(test, embeddings, config)
            row = {
                "held_out_batch": held_out,
                "Method": "FedscGen_two_stage",
                "deployment_mode": "two_stage_correction",
                "n_train_cells": train.n_obs,
                "n_test_cells": test.n_obs,
                "training_time_seconds": training_seconds,
                "correction_time_seconds": correction_seconds,
                "communication_rounds_inference": config["lobo"][
                    "correction_communication_rounds"
                ],
                **metrics,
            }
            rows.append(row)
            write_json(directory / "metrics_two_stage.json", row)
            if config["evaluation"]["save_embeddings"]:
                save_embedding_adata(
                    test,
                    embeddings,
                    directory / "held_out_two_stage_embedding.h5ad",
                    embedding_key="X_fedscgen_two_stage",
                )
            pd.DataFrame(rows).to_csv(summary_path, index=False)
        save_common(directory, config, features, mapping, data_path)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=["multi-dataset", "lobo"])
    parser.add_argument("--config", default="configs/fedscgen.yaml")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--n-hvg", type=int)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pancreas", "immune", "lung"],
        default=["pancreas", "immune", "lung"],
    )
    parser.add_argument("--held-out-batches", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workflow == "multi-dataset":
        output = run_multi(args)
    else:
        output = run_lobo(args)
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
