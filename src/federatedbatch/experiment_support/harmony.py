from __future__ import annotations

import harmonypy

import numpy as np

import pandas as pd

def harmony_embedding(
    pca_embedding: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    result = harmonypy.run_harmony(
        pca_embedding,
        metadata,
        "batch",
        max_iter_harmony=20,
        verbose=False,
        random_state=seed,
    )
    corrected = np.asarray(result.Z_corr)
    if corrected.shape[0] != pca_embedding.shape[0] and corrected.shape[1] == pca_embedding.shape[0]:
        corrected = corrected.T
    if corrected.shape != pca_embedding.shape:
        raise ValueError(
            f"Unexpected Harmony shape for seed {seed}: {corrected.shape}; "
            f"expected {pca_embedding.shape}"
        )
    return corrected
