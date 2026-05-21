"""
Evaluation Metrics for Clustering and Batch Correction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from typing import Dict, Optional
import anndata as ad


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        
    Returns:
        ARI score (higher is better, range [-1, 1])
    """
    return adjusted_rand_score(labels_true, labels_pred)


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        
    Returns:
        NMI score (higher is better, range [0, 1])
    """
    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_asw(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute Average Silhouette Width
    
    Args:
        embeddings: Cell embeddings
        labels: Cell labels
        metric: Distance metric
        
    Returns:
        ASW score (higher is better, range [-1, 1])
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return silhouette_score(embeddings, labels, metric=metric)


def compute_asw_batch(
    embeddings: np.ndarray,
    batch_labels: np.ndarray
) -> float:
    """
    Compute ASW for batch mixing (lower is better for batch correction)
    
    Args:
        embeddings: Cell embeddings
        batch_labels: Batch labels
        
    Returns:
        ASW batch score (lower is better for mixing)
    """
    return compute_asw(embeddings, batch_labels)


def compute_asw_celltype(
    embeddings: np.ndarray,
    celltype_labels: np.ndarray
) -> float:
    """
    Compute ASW for cell type separation (higher is better)
    
    Args:
        embeddings: Cell embeddings
        celltype_labels: Cell type labels
        
    Returns:
        ASW cell type score (higher is better)
    """
    return compute_asw(embeddings, celltype_labels)


def evaluate_clustering(
    adata: ad.AnnData,
    embedding_key: str = 'X_concord',
    celltype_key: str = 'celltype',
    batch_key: str = 'batch',
    cluster_key: Optional[str] = None,
    resolution: float = 1.0
) -> Dict[str, float]:
    """
    Comprehensive evaluation of clustering results
    
    Args:
        adata: AnnData object with embeddings
        embedding_key: Key for embeddings in adata.obsm
        celltype_key: Key for cell type labels
        batch_key: Key for batch labels
        cluster_key: Key for cluster labels (if None, perform clustering)
        resolution: Resolution for Leiden clustering
        
    Returns:
        Dictionary of evaluation metrics
    """
    import scanpy as sc
    
    # Get embeddings
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    embeddings = adata.obsm[embedding_key]
    
    # Perform clustering if needed
    if cluster_key is None or cluster_key not in adata.obs:
        # Compute neighbors and clusters
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
        sc.tl.leiden(adata, resolution=resolution, key_added='clusters')
        cluster_key = 'clusters'
    
    # Get labels
    celltype_labels = adata.obs[celltype_key].values
    batch_labels = adata.obs[batch_key].values
    cluster_labels = adata.obs[cluster_key].values
    
    # Compute metrics
    metrics = {}
    
    # Clustering accuracy
    metrics['ARI'] = compute_ari(celltype_labels, cluster_labels)
    metrics['NMI'] = compute_nmi(celltype_labels, cluster_labels)
    
    # Batch correction (lower is better)
    metrics['ASW_batch'] = compute_asw_batch(embeddings, batch_labels)
    
    # Cell type separation (higher is better)
    metrics['ASW_celltype'] = compute_asw_celltype(embeddings, celltype_labels)
    
    return metrics


def evaluate_batch_correction(
    adata: ad.AnnData,
    embedding_key: str,
    batch_key: str = 'batch'
) -> Dict[str, float]:
    """
    Evaluate batch correction quality
    
    Args:
        adata: AnnData object
        embedding_key: Key for embeddings
        batch_key: Key for batch labels
        
    Returns:
        Dictionary of batch correction metrics
    """
    embeddings = adata.obsm[embedding_key]
    batch_labels = adata.obs[batch_key].values
    
    metrics = {
        'ASW_batch': compute_asw_batch(embeddings, batch_labels),
        'n_batches': len(np.unique(batch_labels))
    }
    
    return metrics


def compare_methods(
    adata: ad.AnnData,
    methods: Dict[str, str],
    celltype_key: str = 'celltype',
    batch_key: str = 'batch'
) -> pd.DataFrame:
    """
    Compare multiple methods
    
    Args:
        adata: AnnData object
        methods: Dictionary mapping method names to embedding keys
        celltype_key: Key for cell type labels
        batch_key: Key for batch labels
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for method_name, embedding_key in methods.items():
        if embedding_key not in adata.obsm:
            print(f"Warning: {embedding_key} not found, skipping {method_name}")
            continue
        
        metrics = evaluate_clustering(
            adata,
            embedding_key=embedding_key,
            celltype_key=celltype_key,
            batch_key=batch_key
        )
        
        metrics['method'] = method_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['method', 'ARI', 'NMI', 'ASW_batch', 'ASW_celltype']]
    
    return df


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print(f"{'='*60}\n")
