"""
Baseline Methods for Comparison

This module exposes baseline methods that do not require external source
trees at import time. CONCORD and FedscGen wrappers can be imported directly
from their modules when their optional dependencies are installed.
"""

from .pca import PCABaseline
from .harmony import HarmonyBaseline

__all__ = [
    "PCABaseline",
    "HarmonyBaseline",
]
