"""FederatedBatch experiment orchestration."""

from .config import load_config, validate_config
from .federated import FederatedResult, train_federated

__all__ = ["FederatedResult", "load_config", "train_federated", "validate_config"]
__version__ = "0.3.0"
