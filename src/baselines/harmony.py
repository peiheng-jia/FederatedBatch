"""
Harmony Baseline

Harmony batch correction method as a traditional baseline.
"""

import numpy as np
from typing import Optional
import anndata as ad


class HarmonyBaseline:
    """
    Harmony baseline for batch correction
    
    Args:
        batch_key: Key for batch information in adata.obs
        max_iter_harmony: Maximum iterations for Harmony
        random_state: Random seed
    """
    
    def __init__(
        self,
        batch_key: str = 'batch',
        max_iter_harmony: int = 10,
        random_state: int = 42
    ):
        self.batch_key = batch_key
        self.max_iter_harmony = max_iter_harmony
        self.random_state = random_state
    
    def fit_transform_adata(
        self,
        adata: ad.AnnData,
        use_rep: str = 'X_pca',
        output_key: str = 'X_harmony'
    ) -> ad.AnnData:
        """
        Apply Harmony batch correction
        
        Args:
            adata: Input AnnData
            use_rep: Representation to use (usually PCA)
            output_key: Key to store results
            
        Returns:
            AnnData with Harmony embeddings
        """
        try:
            import harmonypy as hm
        except ImportError:
            raise ImportError(
                "harmonypy is required for Harmony baseline. "
                "Install it with: pip install harmonypy"
            )
        
        # Get input representation
        if use_rep not in adata.obsm:
            raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
        
        X = adata.obsm[use_rep]
        
        # Get batch information
        if self.batch_key not in adata.obs:
            raise ValueError(f"Batch key '{self.batch_key}' not found in adata.obs")
        
        batch_labels = adata.obs[self.batch_key].values
        
        # Run Harmony
        print(f"Running Harmony batch correction...")
        print(f"  Input: {use_rep} ({X.shape[1]} dimensions)")
        print(f"  Batches: {len(np.unique(batch_labels))}")
        
        ho = hm.run_harmony(
            X,
            adata.obs,
            self.batch_key,
            max_iter_harmony=self.max_iter_harmony,
            random_state=self.random_state
        )
        
        # Store results
        adata.obsm[output_key] = ho.Z_corr.T
        
        print(f"Harmony complete")
        print(f"  Output: {output_key} ({ho.Z_corr.T.shape[1]} dimensions)")
        
        return adata

