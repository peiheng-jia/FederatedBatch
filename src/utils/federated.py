"""
Federated Learning Utilities
"""

import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader, TensorDataset


def federated_averaging(
    client_weights: List[Dict[str, torch.Tensor]],
    client_samples: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Perform FedAvg aggregation
    
    Args:
        client_weights: List of state dicts from clients
        client_samples: Number of samples per client
        
    Returns:
        Aggregated global weights
    """
    total_samples = sum(client_samples)
    global_weights = {}
    
    # Weighted average of parameters
    for key in client_weights[0].keys():
        global_weights[key] = sum(
            client_weights[i][key] * (client_samples[i] / total_samples)
            for i in range(len(client_weights))
        )
    
    return global_weights


def create_client_dataloaders(
    client_data: List,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0
) -> List[DataLoader]:
    """
    Create DataLoaders for each client
    
    Args:
        client_data: List of AnnData objects or arrays
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        List of DataLoaders
    """
    dataloaders = []
    
    for adata_client in client_data:
        # Convert to tensor
        if hasattr(adata_client, 'X'):
            # AnnData object
            X = adata_client.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = torch.FloatTensor(X)
        else:
            # Already an array
            X = torch.FloatTensor(adata_client)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        dataloaders.append(dataloader)
    
    return dataloaders


def distribute_model(
    global_model: torch.nn.Module,
    client_models: List[torch.nn.Module]
):
    """
    Distribute global model weights to all clients
    
    Args:
        global_model: Global model
        client_models: List of client models
    """
    global_weights = global_model.state_dict()
    
    for client_model in client_models:
        client_model.load_state_dict(global_weights)


def compute_communication_cost(
    model: torch.nn.Module,
    n_rounds: int,
    n_clients: int
) -> Dict[str, float]:
    """
    Compute communication cost for federated learning
    
    Args:
        model: Model to analyze
        n_rounds: Number of communication rounds
        n_clients: Number of clients
        
    Returns:
        Dictionary with communication statistics
    """
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    # Assume 32-bit floats (4 bytes per parameter)
    bytes_per_param = 4
    
    # Upload: each client sends weights to server
    upload_per_round = n_params * bytes_per_param * n_clients
    
    # Download: server sends global weights to each client
    download_per_round = n_params * bytes_per_param * n_clients
    
    # Total per round
    total_per_round = upload_per_round + download_per_round
    
    # Total for all rounds
    total_communication = total_per_round * n_rounds
    
    return {
        'n_params': n_params,
        'upload_per_round_MB': upload_per_round / (1024 ** 2),
        'download_per_round_MB': download_per_round / (1024 ** 2),
        'total_per_round_MB': total_per_round / (1024 ** 2),
        'total_communication_MB': total_communication / (1024 ** 2),
        'total_communication_GB': total_communication / (1024 ** 3)
    }


def simulate_client_dropout(
    n_clients: int,
    dropout_rate: float = 0.1,
    random_state: int = 42
) -> List[int]:
    """
    Simulate client dropout in federated learning
    
    Args:
        n_clients: Total number of clients
        dropout_rate: Fraction of clients to drop
        random_state: Random seed
        
    Returns:
        List of active client indices
    """
    np.random.seed(random_state)
    
    n_active = int(n_clients * (1 - dropout_rate))
    active_clients = np.random.choice(n_clients, n_active, replace=False)
    
    return sorted(active_clients.tolist())


def create_non_iid_split(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    random_state: int = 42
) -> List[np.ndarray]:
    """
    Create non-IID data split using Dirichlet distribution
    
    Args:
        labels: Array of labels
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        random_state: Random seed
        
    Returns:
        List of indices for each client
    """
    np.random.seed(random_state)
    
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Initialize client indices
    client_indices = [[] for _ in range(n_clients)]
    
    # For each class, distribute samples to clients using Dirichlet
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        
        # Sample from Dirichlet
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        
        # Split indices
        splits = np.split(label_indices, proportions)
        
        # Assign to clients
        for client_idx, split in enumerate(splits):
            client_indices[client_idx].extend(split.tolist())
    
    # Shuffle each client's data
    for i in range(n_clients):
        np.random.shuffle(client_indices[i])
    
    return [np.array(indices) for indices in client_indices]


def evaluate_data_heterogeneity(
    client_data: List,
    label_key: str = 'celltype'
) -> Dict[str, float]:
    """
    Evaluate data heterogeneity across clients
    
    Args:
        client_data: List of AnnData objects
        label_key: Key for labels
        
    Returns:
        Dictionary with heterogeneity metrics
    """
    # Collect label distributions
    client_distributions = []
    
    for adata in client_data:
        labels = adata.obs[label_key].values
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts / len(labels)))
        client_distributions.append(dist)
    
    # Compute KL divergence between clients
    all_labels = set()
    for dist in client_distributions:
        all_labels.update(dist.keys())
    
    # Average distribution
    avg_dist = {}
    for label in all_labels:
        avg_dist[label] = np.mean([
            dist.get(label, 0) for dist in client_distributions
        ])
    
    # Compute average KL divergence
    kl_divs = []
    for dist in client_distributions:
        kl = 0
        for label in all_labels:
            p = dist.get(label, 1e-10)
            q = avg_dist[label] + 1e-10
            kl += p * np.log(p / q)
        kl_divs.append(kl)
    
    return {
        'avg_kl_divergence': np.mean(kl_divs),
        'max_kl_divergence': np.max(kl_divs),
        'min_kl_divergence': np.min(kl_divs),
        'n_unique_labels': len(all_labels)
    }
