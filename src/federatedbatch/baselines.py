"""Baseline methods used in the manuscript workflows."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .federated import predict_embeddings, resolve_device, set_reproducible_seed


def _dense_features(adata, features: list[str]) -> np.ndarray:
    matrix = adata[:, features].X
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def run_pca(adata, features: list[str], config: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    n_components = min(config["federated"]["latent_dim"], adata.n_obs - 1, len(features))
    model = PCA(n_components=n_components, random_state=config["seed"])
    embeddings = model.fit_transform(_dense_features(adata, features))
    return embeddings, time.perf_counter() - start


def run_pca_zero_shot(train, test, features: list[str], config: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    n_components = min(config["federated"]["latent_dim"], train.n_obs - 1, len(features))
    model = PCA(n_components=n_components, random_state=config["seed"])
    model.fit(_dense_features(train, features))
    embeddings = model.transform(_dense_features(test, features))
    return embeddings, time.perf_counter() - start


def run_harmony(
    adata,
    features: list[str],
    config: dict,
) -> tuple[np.ndarray, float]:
    import harmonypy

    pca_embeddings, _ = run_pca(adata, features, config)
    start = time.perf_counter()
    result = harmonypy.run_harmony(
        pca_embeddings,
        adata.obs,
        "batch",
        max_iter_harmony=20,
        verbose=False,
        random_state=config["seed"],
    )
    corrected = np.asarray(result.Z_corr)
    if corrected.shape[0] != adata.n_obs and corrected.shape[1] == adata.n_obs:
        corrected = corrected.T
    if corrected.shape[0] != adata.n_obs:
        raise ValueError(f"unexpected Harmony output shape: {corrected.shape}")
    return corrected, time.perf_counter() - start


def run_centralized(
    adata,
    features: list[str],
    config: dict,
    output_dir: Path,
) -> tuple[np.ndarray, float]:
    import concord as ccd

    set_reproducible_seed(config["seed"])
    ccd.set_verbose_mode(False)
    fed = config["federated"]
    model = ccd.Concord(
        adata=adata,
        input_feature=features,
        seed=config["seed"],
        device=resolve_device(config["device"]),
        save_dir=output_dir,
        n_epochs=config["centralized"]["epochs"],
        batch_size=fed["batch_size"],
        lr=fed["learning_rate"],
        clr_temperature=fed["temperature"],
        clr_beta=fed["hard_negative_beta"],
        latent_dim=fed["latent_dim"],
        encoder_dims=fed["encoder_dims"],
        dropout_prob=fed["dropout"],
        domain_key="batch",
        preload_dense=True,
        verbose=False,
    )
    model.init_model()
    start = time.perf_counter()
    model.train(save_model=False)
    elapsed = time.perf_counter() - start
    model.init_dataloader(adata=adata, train_frac=1.0, use_sampler=False)
    output = model.predict(
        model.loader,
        return_decoded=False,
        return_class=False,
        return_class_prob=False,
    )
    return np.asarray(output[0]), elapsed


def run_centralized_zero_shot(
    train,
    test,
    features: list[str],
    config: dict,
    output_dir: Path,
) -> tuple[np.ndarray, float]:
    """Train pooled CONCORD without batch-conditioned inputs and encode held-out cells."""
    import concord as ccd

    set_reproducible_seed(config["seed"])
    ccd.set_verbose_mode(False)
    fed = config["federated"]
    model = ccd.Concord(
        adata=train,
        input_feature=features,
        seed=config["seed"],
        device=resolve_device(config["device"]),
        save_dir=output_dir,
        n_epochs=config["centralized"]["epochs"],
        batch_size=fed["batch_size"],
        lr=fed["learning_rate"],
        clr_temperature=fed["temperature"],
        clr_beta=fed["hard_negative_beta"],
        latent_dim=fed["latent_dim"],
        encoder_dims=fed["encoder_dims"],
        dropout_prob=fed["dropout"],
        domain_key=None,
        preload_dense=True,
        verbose=False,
    )
    model.init_model()
    start = time.perf_counter()
    model.train(save_model=False)
    elapsed = time.perf_counter() - start
    return predict_embeddings(model, test), elapsed
