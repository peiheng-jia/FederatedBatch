"""Paper-protocol adapter around the pinned official FedscGen implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "datasets.json"


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"schema_version", "protocol", "seed", "device", "upstream", "training"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"FedscGen configuration is missing: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"unsupported schema version: {config['schema_version']}")
    return config


@lru_cache(maxsize=None)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_upstream(config: dict[str, Any]) -> Path:
    upstream = config["upstream"]
    path = REPO_ROOT / upstream["path"]
    if not (path / "fedscgen" / "FedScGen.py").exists():
        raise FileNotFoundError(
            f"FedscGen vendored source is missing or incomplete at {path}; "
            "restore the documented source snapshot before using the adapter. "
            "This release tree is not a Git submodule checkout."
        )
    if (path / ".git").exists():
        actual_commit = _git_output(path, "rev-parse", "HEAD")
        if actual_commit != upstream["commit"]:
            raise RuntimeError(
                f"FedscGen commit mismatch: expected {upstream['commit']}, got {actual_commit}"
            )
    else:
        manifest_path = Path(__file__).with_name("upstream-source-manifest.json")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest["upstream_commit"] != upstream["commit"]:
            raise RuntimeError("FedscGen source manifest commit does not match the configuration")
        for relative_path, expected_sha256 in manifest["vendored_sha256"].items():
            source_path = path / relative_path
            if not source_path.is_file() or file_sha256(source_path) != expected_sha256:
                raise RuntimeError(f"FedscGen vendored source mismatch: {source_path}")
    patched_source = (path / "fedscgen" / "utils.py").read_text(encoding="utf-8")
    if "Disabled for single-machine setup" not in patched_source:
        raise RuntimeError("FedscGen compatibility patch is not applied; run setup script")
    return path


def import_official(config: dict[str, Any]):
    upstream_path = verify_upstream(config)
    # The pinned upstream's scgen_utils.py uses a top-level ``import __init__``.
    # Support both standalone adapter and canonical-driver entry points.
    sys.path.insert(0, str(upstream_path / "fedscgen"))
    sys.path.insert(0, str(upstream_path))
    from fedscgen.FedScGen import FedScGen
    from fedscgen.utils import aggregate, aggregate_batch_sizes, set_seed

    return FedScGen, aggregate, aggregate_batch_sizes, set_seed


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_dataset(name: str, config: dict[str, Any], verify_hash: bool = True):
    metadata = load_manifest()["datasets"][name]
    path = REPO_ROOT / config["data_dir"] / metadata["filename"]
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.stat().st_size != metadata["size_bytes"]:
        raise ValueError(f"unexpected file size for {name}: {path.stat().st_size}")
    if verify_hash and file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {name}: {path}")

    adata = sc.read_h5ad(path)
    input_layer = metadata.get("input_layer")
    if input_layer:
        if input_layer not in adata.layers:
            raise KeyError(f"required input layer {input_layer!r} is absent")
        values = (
            adata.layers[input_layer].data
            if sparse.issparse(adata.layers[input_layer])
            else np.asarray(adata.layers[input_layer])
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"input layer {input_layer!r} contains invalid values")
        adata.X = adata.layers[input_layer].copy()
    _copy_annotation(
        adata,
        "batch",
        [metadata["batch_key"], *metadata.get("batch_fallbacks", [])],
    )
    _copy_annotation(
        adata,
        "celltype",
        [metadata["cell_type_key"], *metadata.get("cell_type_fallbacks", [])],
    )
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    return adata, path


def _copy_annotation(adata, destination: str, candidates: Iterable[str]) -> None:
    if destination in adata.obs:
        return
    for candidate in candidates:
        if candidate in adata.obs:
            adata.obs[destination] = adata.obs[candidate].copy()
            return
    raise KeyError(f"cannot create obs[{destination!r}] from {list(candidates)}")


def balanced_subsample(adata, max_cells: int, seed: int):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    batches = list(adata.obs["batch"].unique())
    base, remainder = divmod(max_cells, len(batches))
    selected = []
    for index, batch in enumerate(batches):
        batch_indices = np.flatnonzero(np.asarray(adata.obs["batch"] == batch))
        count = min(len(batch_indices), base + (index < remainder))
        selected.extend(rng.choice(batch_indices, size=count, replace=False).tolist())
    if len(selected) < max_cells:
        remaining = np.setdiff1d(np.arange(adata.n_obs), np.asarray(selected))
        extra = rng.choice(remaining, size=min(max_cells - len(selected), len(remaining)), replace=False)
        selected.extend(extra.tolist())
    return adata[np.sort(selected)].copy()


def preprocess(adata, preprocessing: dict[str, Any], n_hvg: int | None = None):
    n_hvg = int(n_hvg or preprocessing["n_hvg"])
    sc.pp.highly_variable_genes(
        adata,
        flavor=preprocessing["hvg_flavor"],
        n_top_genes=n_hvg,
        subset=False,
    )
    features = adata.var_names[adata.var["highly_variable"]].tolist()
    if len(features) != n_hvg:
        raise ValueError(f"expected {n_hvg} HVGs, selected {len(features)}")
    target_sum = preprocessing["normalize_target_sum"]
    if target_sum is None:
        sc.pp.normalize_total(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if preprocessing["log1p"]:
        sc.pp.log1p(adata)
    prepared = adata[:, features].copy()
    prepared.raw = None
    return prepared, features


def seeded_site_groups(adata, n_clients: int, seed: int) -> list[list[str]]:
    batches = np.asarray(list(adata.obs["batch"].unique()), dtype=object)
    if n_clients > len(batches):
        raise ValueError(f"{n_clients} clients requested for {len(batches)} batches")
    random_state = np.random.RandomState(seed)
    random_state.shuffle(batches)
    quotient, remainder = divmod(len(batches), n_clients)
    groups = []
    offset = 0
    for client_index in range(n_clients):
        size = quotient + (client_index < remainder)
        groups.append(batches[offset : offset + size].tolist())
        offset += size
    return groups


def assign_sites(adata, groups: list[list[str]], key: str = "fedscgen_site") -> dict[str, str]:
    mapping = {
        str(batch): f"site_{site_index + 1}"
        for site_index, group in enumerate(groups)
        for batch in group
    }
    unknown = sorted(set(adata.obs["batch"].astype(str)) - mapping.keys())
    if unknown:
        raise ValueError(f"site mapping does not cover batches: {', '.join(unknown)}")
    adata.obs[key] = adata.obs["batch"].astype(str).map(mapping).astype(str)
    return mapping


def _model_kwargs(adata, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    training = config["training"]
    return {
        "ref_model_path": str(output_dir / "reference_model"),
        "adata": adata,
        "hidden_layer_sizes": list(training["hidden_layer_sizes"]),
        "z_dimension": int(training["latent_dim"]),
        "batch_key": "fedscgen_site",
        "cell_key": "celltype",
        "lr": float(training["learning_rate"]),
        "epoch": int(training["local_epochs"]),
        "batch_size": int(training["batch_size"]),
        "stopping": dict(training["early_stopping"]),
        "overwrite": False,
        "device": torch.device(config["device"]),
        "smpc": False,
        "aggregation": training["aggregation"],
        "n_total_samples": adata.n_obs,
    }


def train_official(adata, config: dict[str, Any], output_dir: Path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    FedScGen, aggregate, _, set_seed = import_official(config)
    set_seed(int(config["seed"]))
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {config['device']}")

    training = config["training"]
    site_names = list(adata.obs["fedscgen_site"].unique())
    if len(site_names) != training["n_clients"]:
        raise ValueError(
            f"expected {training['n_clients']} FedscGen sites, found {len(site_names)}"
        )
    kwargs = _model_kwargs(adata, config, output_dir)
    global_model = FedScGen(init_model_path=str(output_dir / "initial_model"), **kwargs)
    global_weights = global_model.model.state_dict()
    clients = []
    for site in site_names:
        client_adata = adata[adata.obs["fedscgen_site"] == site].copy()
        client_kwargs = dict(kwargs)
        client_kwargs["adata"] = client_adata
        clients.append(FedScGen(**client_kwargs))

    start = time.perf_counter()
    for round_index in range(int(training["n_rounds"])):
        state_dicts = []
        sample_counts = []
        for client in clients:
            state, n_samples = client.local_update(global_weights)
            state_dicts.append(state)
            sample_counts.append(n_samples)
        global_weights = aggregate(
            state_dicts,
            sample_counts,
            False,
            list(global_weights.keys()),
            training["aggregation"],
        )
        print(f"FedscGen round {round_index + 1:02d}/{training['n_rounds']}")
    training_seconds = time.perf_counter() - start

    for client in clients:
        client.model.load_state_dict(global_weights)
        client.model.eval()
        client.is_trained_ = True
    global_model.model.load_state_dict(global_weights)
    global_model.model.eval()
    global_model.is_trained_ = True
    global_model.save(str(output_dir / "trained_model"), overwrite=True)
    return global_model, clients, training_seconds


def make_test_client(adata, global_model, config: dict[str, Any], output_dir: Path):
    FedScGen, _, _, _ = import_official(config)
    kwargs = _model_kwargs(adata, config, output_dir)
    client = FedScGen(**kwargs)
    client.model.load_state_dict(global_model.model.state_dict())
    client.model.eval()
    client.is_trained_ = True
    return client


def official_two_stage_correction(clients, config: dict[str, Any]):
    _, _, aggregate_batch_sizes, _ = import_official(config)
    start = time.perf_counter()
    batch_sizes = {
        client_index: client.find_batch_size()
        for client_index, client in enumerate(clients)
    }
    dominant = aggregate_batch_sizes(batch_sizes)
    mean_latent_genes = {}
    for client_index, celltypes in dominant.items():
        mean_latent_genes.update(clients[client_index].avg_local_cells(celltypes))
    corrected = [client.remove_batch_effect(mean_latent_genes) for client in clients]
    return corrected, time.perf_counter() - start


def align_parts(parts: list, obs_names: Iterable[str]):
    combined = ad.concat(parts, join="inner", index_unique=None)
    requested = pd.Index(obs_names)
    if not combined.obs_names.is_unique:
        raise ValueError("FedscGen corrected output contains duplicate cell identifiers")
    missing = requested.difference(combined.obs_names)
    if len(missing):
        raise ValueError(f"FedscGen corrected output is missing {len(missing)} cells")
    return combined[requested].copy()


def latent_embeddings(client, source) -> np.ndarray:
    latent = client.get_latent_adata(source).X
    if torch.is_tensor(latent):
        latent = latent.detach().cpu().numpy()
    return np.asarray(latent, dtype=np.float32)


def save_embedding_adata(
    source,
    embeddings: np.ndarray,
    path: Path,
    *,
    embedding_key: str = "X_fedscgen",
) -> None:
    result = ad.AnnData(obs=source.obs.copy())
    result.obsm[embedding_key] = np.asarray(embeddings, dtype=np.float32)
    result.write_h5ad(path, compression="gzip")


def evaluate(source, embeddings: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from federatedbatch.evaluation import evaluate_embedding

    eval_config = {"seed": config["seed"], "evaluation": config["evaluation"]}
    return evaluate_embedding(
        source,
        np.asarray(embeddings),
        eval_config,
        embedding_key="X_fedscgen",
    )


def make_run_dir(
    config: dict[str, Any],
    experiment: str,
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    if output_dir is None:
        root = REPO_ROOT / config["output_dir"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / f"fedscgen_{experiment}_{timestamp}"
    else:
        output = Path(output_dir)
        if not output.is_absolute():
            output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=resume)
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(value),
            handle,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        return None
    return value


def provenance(config: dict[str, Any], data_path: Path, features: list[str]) -> dict[str, Any]:
    packages = {}
    for name in ("anndata", "scanpy", "scarches", "scgen", "scvi-tools", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    patch_path = REPO_ROOT / config["upstream"]["patch"]
    return {
        "protocol": config["protocol"],
        "repository_commit": _git_output(REPO_ROOT, "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "upstream_commit": config["upstream"]["commit"],
        "compatibility_patch_sha256": file_sha256(patch_path),
        "data": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
        },
        "features_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
    }
