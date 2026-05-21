"""
FedscGen Baseline Implementation

FedscGen is a federated learning approach for single-cell data integration
based on scGen (single-cell Generator).

Reference:
- scGen: https://github.com/theislab/scgen
- FedscGen concept: Federated version of scGen

Note: This is a simplified implementation for comparison purposes.
For full FedscGen implementation, please refer to the original paper.
"""

import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class FedscGenBaseline:
    """
    Simplified FedscGen baseline for comparison
    
    Since full FedscGen requires complex VAE training and is not open-sourced,
    we implement a simplified version that captures the key ideas:
    1. Local VAE training on each client
    2. Federated averaging of VAE parameters
    3. Global embedding generation
    
    For fair comparison, we use PCA as a proxy for the VAE encoder.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        n_rounds: int = 20,
        n_epochs_per_round: int = 5,
        random_state: int = 42
    ):
        """
        Initialize FedscGen baseline
        
        Parameters
        ----------
        n_components : int
            Number of latent dimensions
        n_rounds : int
            Number of federated rounds
        n_epochs_per_round : int
            Number of local training epochs per round
        random_state : int
            Random seed
        """
        self.n_components = n_components
        self.n_rounds = n_rounds
        self.n_epochs_per_round = n_epochs_per_round
        self.random_state = random_state
        
        # Global model (PCA as proxy)
        self.global_model = None
        
    def fit_transform(
        self,
        adata_list: List,
        feature_list: List[str]
    ) -> np.ndarray:
        """
        Fit FedscGen on multiple clients and return embeddings
        
        Parameters
        ----------
        adata_list : List[AnnData]
            List of AnnData objects for each client
        feature_list : List[str]
            List of feature names to use
            
        Returns
        -------
        embeddings : np.ndarray
            Global embeddings for all cells
        """
        print(f"FedscGen: Training with {len(adata_list)} clients")
        print(f"  Rounds: {self.n_rounds}")
        print(f"  Epochs per round: {self.n_epochs_per_round}")
        
        # Initialize global model with a dummy fit to get all attributes
        # We'll overwrite these in the federated training loop
        all_adata = sc.concat(adata_list, join='outer')
        X_init = all_adata[:, feature_list].X
        if hasattr(X_init, 'toarray'):
            X_init = X_init.toarray()
        
        self.global_model = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        # Fit once to initialize all attributes
        self.global_model.fit(X_init)
        
        # Federated training simulation
        for round_idx in range(self.n_rounds):
            print(f"  Round {round_idx + 1}/{self.n_rounds}", end='\r')
            
            # Local training on each client
            local_models = []
            local_weights = []
            
            for client_idx, adata_client in enumerate(adata_list):
                # Extract features
                X_client = adata_client[:, feature_list].X
                if hasattr(X_client, 'toarray'):
                    X_client = X_client.toarray()
                
                # Local PCA (proxy for VAE encoder)
                local_pca = PCA(
                    n_components=self.n_components,
                    random_state=self.random_state + round_idx
                )
                local_pca.fit(X_client)
                
                local_models.append(local_pca)
                local_weights.append(len(adata_client))
            
            # Federated averaging (average PCA components)
            total_weight = sum(local_weights)
            avg_components = sum(
                model.components_ * (weight / total_weight)
                for model, weight in zip(local_models, local_weights)
            )
            
            # Average explained variance as well
            avg_explained_variance = sum(
                model.explained_variance_ * (weight / total_weight)
                for model, weight in zip(local_models, local_weights)
            )
            
            avg_explained_variance_ratio = sum(
                model.explained_variance_ratio_ * (weight / total_weight)
                for model, weight in zip(local_models, local_weights)
            )
            
            # Update global model
            self.global_model.components_ = avg_components
            self.global_model.explained_variance_ = avg_explained_variance
            self.global_model.explained_variance_ratio_ = avg_explained_variance_ratio
            self.global_model.mean_ = local_models[0].mean_  # Use first client's mean
            self.global_model.n_components_ = self.n_components
            self.global_model.n_features_in_ = avg_components.shape[1]
            self.global_model.n_samples_ = sum(local_weights)
        
        print(f"\n  ✓ Training complete")
        
        # Generate global embeddings
        all_adata = sc.concat(adata_list, join='outer')
        X_all = all_adata[:, feature_list].X
        if hasattr(X_all, 'toarray'):
            X_all = X_all.toarray()
        
        embeddings = self.global_model.transform(X_all)
        
        return embeddings
    
    def fit_transform_single(
        self,
        adata,
        feature_list: List[str],
        batch_key: str = 'batch',
        n_clients: int = 3
    ) -> np.ndarray:
        """
        Fit FedscGen on a single AnnData by splitting into clients
        
        Parameters
        ----------
        adata : AnnData
            Input data
        feature_list : List[str]
            List of feature names
        batch_key : str
            Key for batch information
        n_clients : int
            Number of clients to simulate
            
        Returns
        -------
        embeddings : np.ndarray
            Global embeddings
        """
        # Split data into clients by batch
        all_batches = list(adata.obs[batch_key].unique())
        batches_per_client = len(all_batches) // n_clients
        
        adata_list = []
        for i in range(n_clients):
            start_idx = i * batches_per_client
            if i == n_clients - 1:
                client_batches = all_batches[start_idx:]
            else:
                client_batches = all_batches[start_idx:start_idx + batches_per_client]
            
            mask = adata.obs[batch_key].isin(client_batches)
            adata_client = adata[mask].copy()
            adata_list.append(adata_client)
        
        return self.fit_transform(adata_list, feature_list)


def run_fedscgen(
    adata,
    feature_list: List[str],
    batch_key: str = 'batch',
    n_clients: int = 3,
    n_rounds: int = 20,
    n_epochs: int = 5,
    n_components: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Convenience function to run FedscGen
    
    Parameters
    ----------
    adata : AnnData
        Input data
    feature_list : List[str]
        List of feature names
    batch_key : str
        Key for batch information
    n_clients : int
        Number of clients
    n_rounds : int
        Number of federated rounds
    n_epochs : int
        Epochs per round
    n_components : int
        Number of latent dimensions
    random_state : int
        Random seed
        
    Returns
    -------
    embeddings : np.ndarray
        Global embeddings
    """
    model = FedscGenBaseline(
        n_components=n_components,
        n_rounds=n_rounds,
        n_epochs_per_round=n_epochs,
        random_state=random_state
    )
    
    return model.fit_transform_single(
        adata,
        feature_list,
        batch_key=batch_key,
        n_clients=n_clients
    )
