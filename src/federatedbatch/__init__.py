"""
FederatedBatch: Federated Contrastive Learning for Single-Cell Data Integration

A federated learning framework for single-cell RNA-seq data integration that
preserves data privacy while achieving performance comparable to centralized methods.
"""

__version__ = "1.0.0"
__author__ = "FederatedBatch Team"

from .model import FederatedBatch

__all__ = [
    "FederatedBatch",
]
