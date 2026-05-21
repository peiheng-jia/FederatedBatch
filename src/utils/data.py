"""
Data Loading and Preprocessing Utilities
"""

import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import anndata as ad


def load_pancreas_data(data_path: str = "data/pancreas.h5ad") -> ad.AnnData:
    """
    Load the pancreas dataset
    
    Args:
        data_path: Path to the pancreas.h5ad file
        
    Returns:
        AnnData object with pancreas data
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            f"Please run scripts/download_data.py first."
        )
    
    adata = sc.read_h5ad(data_path)
    
    # Standardize column names
    if 'tech' in adata.obs.columns and 'batch' not in adata.obs.columns:
        adata.obs['batch'] = adata.obs['tech']
    
    if 'celltype' not in adata.obs.columns:
        for col in ['cell_type', 'louvain', 'leiden']:
            if col in adata.obs.columns:
                adata.obs['celltype'] = adata.obs[col]
                break
    
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  Batches: {len(adata.obs['batch'].unique())}")
    print(f"  Cell types: {len(adata.obs['celltype'].unique())}")
    
    return adata


def preprocess_data(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    target_sum: float = 1e4,
    log_transform: bool = True
) -> Tuple[ad.AnnData, List[str]]:
    """
    Preprocess single-cell data
    
    Args:
        adata: Input AnnData object
        n_top_genes: Number of highly variable genes to select
        target_sum: Target sum for normalization
        log_transform: Whether to apply log1p transformation
        
    Returns:
        Preprocessed AnnData and list of selected features
    """
    # Select highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        flavor='seurat_v3',
        n_top_genes=n_top_genes
    )
    feature_list = adata.var_names[adata.var.highly_variable].tolist()
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
    
    print(f"Selected {len(feature_list)} highly variable genes")
    print(f"Normalized to {target_sum} total counts")
    if log_transform:
        print(f"Applied log1p transformation")
    
    return adata, feature_list


def partition_data(
    adata: ad.AnnData,
    batch_key: str = 'batch',
    n_clients: Optional[int] = None
) -> List[ad.AnnData]:
    """
    Partition data into clients based on batch
    
    Args:
        adata: Input AnnData object
        batch_key: Column name for batch information
        n_clients: Number of clients (if None, use all batches)
        
    Returns:
        List of AnnData objects, one per client
    """
    batches = adata.obs[batch_key].unique()
    
    if n_clients is not None and n_clients < len(batches):
        # Randomly select n_clients batches
        np.random.shuffle(batches)
        batches = batches[:n_clients]
    
    client_data = []
    for i, batch in enumerate(batches):
        mask = adata.obs[batch_key] == batch
        adata_client = adata[mask].copy()
        client_data.append(adata_client)
        print(f"  Client {i}: {batch} ({len(adata_client)} cells)")
    
    print(f"Partitioned into {len(client_data)} clients")
    
    return client_data


def create_train_val_split(
    adata: ad.AnnData,
    train_frac: float = 0.8,
    random_state: int = 42
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Split data into training and validation sets
    
    Args:
        adata: Input AnnData object
        train_frac: Fraction of data for training
        random_state: Random seed
        
    Returns:
        Training and validation AnnData objects
    """
    np.random.seed(random_state)
    n_obs = adata.n_obs
    n_train = int(n_obs * train_frac)
    
    indices = np.random.permutation(n_obs)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    adata_train = adata[train_indices].copy()
    adata_val = adata[val_indices].copy()
    
    print(f"Split: {len(adata_train)} train, {len(adata_val)} val")
    
    return adata_train, adata_val


def subsample_cells(
    adata: ad.AnnData,
    n_cells: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42
) -> ad.AnnData:
    """
    Subsample cells from AnnData
    
    Args:
        adata: Input AnnData object
        n_cells: Number of cells to sample (if None, use frac)
        frac: Fraction of cells to sample
        random_state: Random seed
        
    Returns:
        Subsampled AnnData object
    """
    np.random.seed(random_state)
    
    if n_cells is not None:
        n_cells = min(n_cells, adata.n_obs)
    elif frac is not None:
        n_cells = int(adata.n_obs * frac)
    else:
        return adata
    
    indices = np.random.choice(adata.n_obs, n_cells, replace=False)
    adata_sub = adata[indices].copy()
    
    print(f"Subsampled {len(adata_sub)} cells from {adata.n_obs}")
    
    return adata_sub


def filter_genes(
    adata: ad.AnnData,
    min_cells: int = 3,
    min_counts: int = 1
) -> ad.AnnData:
    """
    Filter genes based on expression
    
    Args:
        adata: Input AnnData object
        min_cells: Minimum number of cells expressing a gene
        min_counts: Minimum total counts for a gene
        
    Returns:
        Filtered AnnData object
    """
    n_genes_before = adata.n_vars
    
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    
    n_genes_after = adata.n_vars
    print(f"Filtered genes: {n_genes_before} ->{n_genes_after}")
    
    return adata


def filter_cells(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_counts: int = 1000
) -> ad.AnnData:
    """
    Filter cells based on quality
    
    Args:
        adata: Input AnnData object
        min_genes: Minimum number of genes expressed
        min_counts: Minimum total counts
        
    Returns:
        Filtered AnnData object
    """
    n_cells_before = adata.n_obs
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    
    n_cells_after = adata.n_obs
    print(f"Filtered cells: {n_cells_before} ->{n_cells_after}")
    
    return adata

