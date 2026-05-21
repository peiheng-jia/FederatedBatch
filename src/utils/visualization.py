"""
Visualization Utilities for Single-Cell Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import anndata as ad


def plot_umap(
    adata: ad.AnnData,
    embedding_key: str,
    color_by: str = 'celltype',
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    **kwargs
):
    """
    Plot UMAP visualization
    
    Args:
        adata: AnnData object
        embedding_key: Key for embeddings in adata.obsm
        color_by: Column to color points by
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        **kwargs: Additional arguments for scanpy.pl.embedding
    """
    import scanpy as sc
    
    # Compute UMAP if not already present
    umap_key = f'{embedding_key}_umap'
    if umap_key not in adata.obsm:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
        sc.tl.umap(adata)
        adata.obsm[umap_key] = adata.obsm['X_umap'].copy()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sc.pl.embedding(
        adata,
        basis=umap_key,
        color=color_by,
        title=title or f'UMAP - {embedding_key}',
        ax=ax,
        show=False,
        **kwargs
    )
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved UMAP to {save_path}")
    
    plt.close()


def plot_comparison_umap(
    adata: ad.AnnData,
    methods: Dict[str, str],
    color_by: str = 'celltype',
    ncols: int = 3,
    figsize: tuple = (15, 10),
    save_path: Optional[Path] = None
):
    """
    Plot UMAP comparison for multiple methods
    
    Args:
        adata: AnnData object
        methods: Dictionary mapping method names to embedding keys
        color_by: Column to color points by
        ncols: Number of columns in subplot grid
        figsize: Figure size
        save_path: Path to save figure
    """
    import scanpy as sc
    
    n_methods = len(methods)
    nrows = (n_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    for idx, (method_name, embedding_key) in enumerate(methods.items()):
        if embedding_key not in adata.obsm:
            print(f"Warning: {embedding_key} not found, skipping")
            continue
        
        # Compute UMAP
        umap_key = f'{embedding_key}_umap'
        if umap_key not in adata.obsm:
            sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=15)
            sc.tl.umap(adata)
            adata.obsm[umap_key] = adata.obsm['X_umap'].copy()
        
        # Plot
        sc.pl.embedding(
            adata,
            basis=umap_key,
            color=color_by,
            title=method_name,
            ax=axes[idx],
            show=False
        )
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison UMAP to {save_path}")
    
    plt.close()


def plot_metrics(
    results_df: pd.DataFrame,
    metrics: List[str] = ['ARI', 'NMI', 'ASW_batch', 'ASW_celltype'],
    figsize: tuple = (12, 4),
    save_path: Optional[Path] = None
):
    """
    Plot comparison of metrics across methods
    
    Args:
        results_df: DataFrame with method comparison results
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in results_df.columns:
            continue
        
        ax = axes[idx]
        
        # Bar plot
        results_df.plot(
            x='method',
            y=metric,
            kind='bar',
            ax=ax,
            legend=False,
            color='steelblue'
        )
        
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics plot to {save_path}")
    
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = ['train_loss', 'val_loss'],
    figsize: tuple = (10, 4),
    save_path: Optional[Path] = None
):
    """
    Plot training curves
    
    Args:
        history: Dictionary of training history
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in history:
            continue
        
        ax = axes[idx]
        values = history[metric]
        
        ax.plot(values, marker='o', linewidth=2)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Round' if 'round' in metric.lower() else 'Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    plt.close()


def plot_ablation_results(
    results_df: pd.DataFrame,
    param_name: str,
    metric: str = 'ARI',
    figsize: tuple = (8, 5),
    save_path: Optional[Path] = None
):
    """
    Plot ablation study results
    
    Args:
        results_df: DataFrame with ablation results
        param_name: Name of the parameter being ablated
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Line plot with markers
    ax.plot(
        results_df[param_name],
        results_df[metric],
        marker='o',
        linewidth=2,
        markersize=8,
        color='steelblue'
    )
    
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs {param_name.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(results_df[param_name], results_df[metric]):
        ax.annotate(
            f'{y:.3f}',
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ablation plot to {save_path}")
    
    plt.close()


def plot_heatmap(
    data: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Heatmap",
    figsize: tuple = (10, 8),
    cmap: str = 'viridis',
    save_path: Optional[Path] = None
):
    """
    Plot heatmap
    
    Args:
        data: 2D array to plot
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': 'Value'}
    )
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {save_path}")
    
    plt.close()
