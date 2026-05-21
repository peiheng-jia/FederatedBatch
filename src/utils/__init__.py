"""
Utility Functions

This module provides utility functions for:
- Data loading and preprocessing
- Evaluation metrics (ARI, NMI, ASW)
- Visualization (UMAP, t-SNE)
- Federated learning utilities
"""

from .data import load_pancreas_data, partition_data, preprocess_data
from .metrics import evaluate_clustering, compute_ari, compute_nmi, compute_asw
from .visualization import plot_umap, plot_metrics, plot_training_curves
from .federated import federated_averaging, create_client_dataloaders

__all__ = [
    # Data
    "load_pancreas_data",
    "partition_data",
    "preprocess_data",
    # Metrics
    "evaluate_clustering",
    "compute_ari",
    "compute_nmi",
    "compute_asw",
    # Visualization
    "plot_umap",
    "plot_metrics",
    "plot_training_curves",
    # Federated
    "federated_averaging",
    "create_client_dataloaders",
]
