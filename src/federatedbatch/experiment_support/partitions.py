from __future__ import annotations

import copy

import hashlib

import numpy as np

import scanpy as sc

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from federatedbatch.partition import partition_cells

METHODS = ("NoQueue", "FreshQueue")

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
