"""
FederatedBatch Model

Main model class for federated contrastive learning on single-cell data.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import copy


class FederatedBatch:
    """
    FederatedBatch: Federated Contrastive Learning for Single-Cell Data Integration
    
    This class implements the federated learning framework for single-cell RNA-seq
    data integration using contrastive learning with a global queue mechanism.
    
    Args:
        n_clients (int): Number of federated clients
        input_dim (int): Input feature dimension
        latent_dim (int): Latent embedding dimension
        encoder_dims (List[int]): Hidden layer dimensions for encoder
        n_rounds (int): Number of federated learning rounds
        local_epochs (int): Number of local training epochs per round
        batch_size (int): Batch size for training
        lr (float): Learning rate
        temperature (float): Temperature parameter for contrastive loss
        queue_size (int): Maximum size of global queue
        device (torch.device): Device to run on (CPU or CUDA)
        **kwargs: Additional configuration parameters
    """
    
    def __init__(
        self,
        n_clients: int,
        input_dim: int,
        latent_dim: int = 100,
        encoder_dims: List[int] = [128],
        n_rounds: int = 10,
        local_epochs: int = 2,
        batch_size: int = 128,
        lr: float = 1e-3,
        temperature: float = 0.07,
        queue_size: int = 1000,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        self.n_clients = n_clients
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.queue_size = queue_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Additional parameters
        self.dropout_prob = kwargs.get('dropout_prob', 0.1)
        self.domain_key = kwargs.get('domain_key', 'batch')
        self.verbose = kwargs.get('verbose', True)
        
        # Global model and queue
        self.global_model = None
        self.global_queue = []
        
        # Client models
        self.client_models = []
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'queue_size': []
        }
    
    def _build_encoder(self) -> nn.Module:
        """Build the encoder network"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob)
            ])
            prev_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def initialize_clients(self, client_data: List):
        """
        Initialize client models with their local data
        
        Args:
            client_data: List of AnnData objects, one per client
        """
        self.client_models = []
        
        for i in range(self.n_clients):
            # Each client gets a copy of the encoder
            model = self._build_encoder().to(self.device)
            self.client_models.append(model)
        
        # Initialize global model
        self.global_model = self._build_encoder().to(self.device)
        
        if self.verbose:
            print(f"Initialized {self.n_clients} client models")
            total_params = sum(p.numel() for p in self.global_model.parameters())
            print(f"Total parameters: {total_params:,}")
    
    def local_train(self, client_idx: int, dataloader, optimizer):
        """
        Perform local training on a single client
        
        Args:
            client_idx: Index of the client
            dataloader: DataLoader for client's local data
            optimizer: Optimizer for local training
            
        Returns:
            Average training loss
        """
        model = self.client_models[client_idx]
        model.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            # Get data
            if isinstance(batch, dict):
                x = batch['input'].to(self.device)
            else:
                x = batch[0].to(self.device)
            
            # Two stochastic views are produced by dropout in the encoder.
            z1 = model(x)
            z2 = model(x)
            
            # Compute contrastive loss
            loss = self._compute_contrastive_loss(z1, z2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def _compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss with global queue
        
        Args:
            embeddings: Batch of embeddings (B, D)
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        positive_embeddings = nn.functional.normalize(positive_embeddings, dim=1)
        
        # The matching row in positive_embeddings is the positive target.
        sim_matrix = torch.matmul(embeddings, positive_embeddings.T) / self.temperature
        
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size).to(self.device)
        
        # Add global queue negatives if available
        if len(self.global_queue) > 0:
            queue_tensor = torch.tensor(
                np.array(self.global_queue), 
                dtype=torch.float32
            ).to(self.device)
            queue_tensor = nn.functional.normalize(queue_tensor, dim=1)
            
            # Compute similarity with queue
            queue_sim = torch.matmul(embeddings, queue_tensor.T) / self.temperature
            sim_matrix = torch.cat([sim_matrix, queue_sim], dim=1)
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def aggregate_weights(self, client_weights: List[Dict], client_samples: List[int]):
        """
        Aggregate client weights using FedAvg
        
        Args:
            client_weights: List of state dicts from clients
            client_samples: Number of samples per client
        """
        total_samples = sum(client_samples)
        global_weights = {}
        
        # Weighted average of parameters
        for key in client_weights[0].keys():
            global_weights[key] = sum(
                client_weights[i][key] * (client_samples[i] / total_samples)
                for i in range(len(client_weights))
            )
        
        # Update global model
        self.global_model.load_state_dict(global_weights)
        
        # Distribute to clients
        for model in self.client_models:
            model.load_state_dict(global_weights)
    
    def update_global_queue(self, new_embeddings: np.ndarray):
        """
        Update global queue with new embeddings
        
        Args:
            new_embeddings: New embeddings to add to queue
        """
        # Add new embeddings
        self.global_queue.extend(new_embeddings.tolist())
        
        # Keep only most recent embeddings (FIFO)
        if len(self.global_queue) > self.queue_size:
            self.global_queue = self.global_queue[-self.queue_size:]
    
    def fit(self, client_dataloaders: List, client_samples: List[int]):
        """
        Train the federated model
        
        Args:
            client_dataloaders: List of DataLoaders, one per client
            client_samples: Number of samples per client
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting Federated Training")
            print(f"{'='*80}")
            print(f"Clients: {self.n_clients}")
            print(f"Rounds: {self.n_rounds}")
            print(f"Local epochs: {self.local_epochs}")
            print(f"Queue size: {self.queue_size}")
        
        for round_idx in range(self.n_rounds):
            if self.verbose:
                print(f"\nRound {round_idx + 1}/{self.n_rounds}")
            
            # Local training
            local_weights = []
            round_losses = []
            
            for client_idx in range(self.n_clients):
                # Create optimizer for this client
                optimizer = torch.optim.Adam(
                    self.client_models[client_idx].parameters(),
                    lr=self.lr
                )
                
                # Train for local epochs
                client_loss = 0.0
                for epoch in range(self.local_epochs):
                    loss = self.local_train(
                        client_idx,
                        client_dataloaders[client_idx],
                        optimizer
                    )
                    client_loss += loss
                
                avg_client_loss = client_loss / self.local_epochs
                round_losses.append(avg_client_loss)
                
                # Collect weights
                local_weights.append(self.client_models[client_idx].state_dict())
                
                # Collect embeddings for queue
                embeddings = self._extract_embeddings(
                    client_idx,
                    client_dataloaders[client_idx]
                )
                self.update_global_queue(embeddings[:100])  # Add top 100
            
            # Aggregate weights
            self.aggregate_weights(local_weights, client_samples)
            
            # Record history
            avg_round_loss = np.mean(round_losses)
            self.history['train_loss'].append(avg_round_loss)
            self.history['queue_size'].append(len(self.global_queue))
            
            if self.verbose:
                print(f"  Avg Loss: {avg_round_loss:.4f}")
                print(f"  Queue Size: {len(self.global_queue)}")
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Training Complete")
            print(f"{'='*80}")
    
    def _extract_embeddings(self, client_idx: int, dataloader) -> np.ndarray:
        """Extract embeddings from a client's data"""
        model = self.client_models[client_idx]
        model.eval()
        
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['input'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                z = model(x)
                embeddings.append(z.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def transform(self, dataloader) -> np.ndarray:
        """
        Transform data using the global model
        
        Args:
            dataloader: DataLoader for the data to transform
            
        Returns:
            Embeddings as numpy array
        """
        self.global_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['input'].to(self.device)
                else:
                    x = batch[0].to(self.device)
                
                z = self.global_model(x)
                embeddings.append(z.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def save(self, save_path: Path):
        """Save the global model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'encoder_dims': self.encoder_dims,
                'n_clients': self.n_clients,
                'queue_size': self.queue_size,
                'temperature': self.temperature
            },
            'history': self.history
        }, save_path)
        
        if self.verbose:
            print(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Path, device: Optional[torch.device] = None):
        """Load a saved model"""
        checkpoint = torch.load(load_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create instance
        model = cls(
            n_clients=config['n_clients'],
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            encoder_dims=config['encoder_dims'],
            queue_size=config['queue_size'],
            temperature=config['temperature'],
            device=device
        )
        
        # Load weights
        model.global_model = model._build_encoder().to(model.device)
        model.global_model.load_state_dict(checkpoint['model_state_dict'])
        model.history = checkpoint.get('history', {})
        
        return model

