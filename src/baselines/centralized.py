"""
Centralized CONCORD Baseline

Full CONCORD model trained on centralized data (upper bound).
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Optional, List
import anndata as ad

# Import CONCORD from the original implementation
try:
    concord_path = Path(__file__).parent.parent.parent.parent / 'Concord-main' / 'src'
    if str(concord_path) not in sys.path:
        sys.path.insert(0, str(concord_path))
    import concord as ccd
except ImportError:
    raise ImportError(
        "CONCORD package not found. Please ensure Concord-main/src is in the path."
    )


class CentralizedCONCORD:
    """
    Centralized CONCORD baseline (upper bound)
    
    This serves as the performance upper bound by training on all data centrally.
    
    Args:
        input_feature: List of feature names to use
        latent_dim: Latent embedding dimension
        encoder_dims: Hidden layer dimensions
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        temperature: Temperature for contrastive loss
        domain_key: Key for batch/domain information
        device: Device to run on
        **kwargs: Additional CONCORD parameters
    """
    
    def __init__(
        self,
        input_feature: Optional[List[str]] = None,
        latent_dim: int = 100,
        encoder_dims: List[int] = [128],
        n_epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        temperature: float = 0.07,
        domain_key: str = 'batch',
        device: Optional[torch.device] = None,
        **kwargs
    ):
        self.input_feature = input_feature
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.domain_key = domain_key
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.kwargs = kwargs
        
        self.model = None
    
    def fit(
        self,
        adata: ad.AnnData,
        save_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Train centralized CONCORD model
        
        Args:
            adata: Input AnnData with all data
            save_dir: Directory to save model
            verbose: Whether to print progress
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training Centralized CONCORD")
            print(f"{'='*80}")
            print(f"Data: {adata.n_obs} cells, {adata.n_vars} genes")
            print(f"Batches: {len(adata.obs[self.domain_key].unique())}")
        
        # Create CONCORD model
        self.model = ccd.Concord(
            adata=adata,
            input_feature=self.input_feature,
            latent_dim=self.latent_dim,
            encoder_dims=self.encoder_dims,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            clr_temperature=self.temperature,
            domain_key=self.domain_key,
            device=self.device,
            save_dir=save_dir,
            verbose=verbose,
            preload_dense=True,
            **self.kwargs
        )
        
        # Initialize and train
        self.model.init_model()
        self.model.train(save_model=False)
        
        if verbose:
            print(f"✓ Training complete")
    
    def transform(
        self,
        adata: ad.AnnData,
        output_key: str = 'X_concord'
    ) -> np.ndarray:
        """
        Transform data using trained model
        
        Args:
            adata: Input AnnData
            output_key: Key to store embeddings
            
        Returns:
            Embeddings
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before transform")
        
        # Initialize dataloader
        self.model.init_dataloader(adata=adata, train_frac=1.0, use_sampler=False)
        
        # Get embeddings
        results = self.model.predict(
            self.model.loader,
            return_decoded=False,
            return_class=False,
            return_class_prob=False
        )
        
        embeddings = results[0]
        
        # Store in adata
        adata.obsm[output_key] = embeddings
        
        return embeddings
    
    def fit_transform(
        self,
        adata: ad.AnnData,
        output_key: str = 'X_concord',
        save_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Fit and transform data
        
        Args:
            adata: Input AnnData
            output_key: Key to store embeddings
            save_dir: Directory to save model
            verbose: Whether to print progress
            
        Returns:
            Embeddings
        """
        self.fit(adata, save_dir=save_dir, verbose=verbose)
        return self.transform(adata, output_key=output_key)
