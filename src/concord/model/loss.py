

import torch
from torch import nn
from torch.nn import functional as F
import math
    

class NTXent_general(nn.Module):
    def __init__(self, temperature=0.5, beta=0.0):
        super().__init__()
        self.T, self.beta = temperature, beta

    def forward(self, z_i, z_j):
        B = z_i.size(0)
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)     # 2B × d
        logits = z @ z.T / self.T                                # 2B × 2B
        pos_idx = (torch.arange(2*B, device=z.device) + B) % (2*B)

        # -------- hard-negative weighting --------
        if self.beta > 0:
            neg_mask = torch.ones_like(logits, dtype=torch.bool)
            neg_mask.fill_diagonal_(False)
            neg_mask[torch.arange(2*B, device=z.device), pos_idx] = False

            neg_logits = logits[neg_mask].view(2*B, -1)                # (2B,2B-2)
            m = neg_logits.size(1)                                     # 2B-2
            log_Z_beta = (torch.logsumexp(self.beta * neg_logits, 1, keepdim=True)
                          - math.log(m))                               # average
            neg_logits = (self.beta + 1) * neg_logits - log_Z_beta
            logits[neg_mask] = neg_logits.reshape(-1)

        # -------- cross-entropy NT-Xent --------
        logits.fill_diagonal_(float('-inf'))
        loss = F.cross_entropy(logits, pos_idx)
        return loss


def importance_penalty_loss(importance_weights, penalty_weight=0.1, norm_type='L1'):
    if penalty_weight == 0:
        return torch.tensor(0.0, device=importance_weights.device)

    input_dim = importance_weights.size(0)
    if norm_type == 'L1':
        base_penalty_weight = 1 / input_dim
        penalty_loss = base_penalty_weight * torch.norm(importance_weights, p=1)
    elif norm_type == 'L2':
        base_penalty_weight = 1 / torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
        penalty_loss = base_penalty_weight * torch.norm(importance_weights, p=2)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    return penalty_weight * penalty_loss



class NTXent_with_queue(nn.Module):
    """
    NT-Xent loss with global queue support for federated learning.
    
    This extends the standard NT-Xent loss to include embeddings from a global queue
    as additional negative samples, enabling cross-client contrastive learning.
    
    Args:
        temperature (float): Temperature parameter for softmax. Default: 0.5
        beta (float): Hard negative weighting parameter. Default: 0.0
        queue_weight (float): Weight for queue samples vs batch samples. Default: 1.0
    """
    def __init__(self, temperature=0.5, beta=0.0, queue_weight=1.0):
        super().__init__()
        self.T = temperature
        self.beta = beta
        self.queue_weight = queue_weight

    def forward(self, z_i, z_j, global_queue=None):
        """
        Compute NT-Xent loss with optional global queue.
        
        Args:
            z_i (torch.Tensor): First view embeddings, shape (B, d)
            z_j (torch.Tensor): Second view embeddings, shape (B, d)
            global_queue (torch.Tensor, optional): Global queue embeddings, shape (Q, d)
        
        Returns:
            torch.Tensor: Contrastive loss value
        """
        B = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        if global_queue is None or len(global_queue) == 0:
            # Standard NT-Xent without queue
            z = torch.cat([z_i, z_j], dim=0)  # 2B × d
            logits = z @ z.T / self.T          # 2B × 2B
            pos_idx = (torch.arange(2*B, device=z.device) + B) % (2*B)
            
            # Hard-negative weighting
            if self.beta > 0:
                neg_mask = torch.ones_like(logits, dtype=torch.bool)
                neg_mask.fill_diagonal_(False)
                neg_mask[torch.arange(2*B, device=z.device), pos_idx] = False
                
                neg_logits = logits[neg_mask].view(2*B, -1)
                m = neg_logits.size(1)
                log_Z_beta = (torch.logsumexp(self.beta * neg_logits, 1, keepdim=True)
                              - math.log(m))
                neg_logits = (self.beta + 1) * neg_logits - log_Z_beta
                logits[neg_mask] = neg_logits.reshape(-1)
            
            # Cross-entropy NT-Xent
            logits.fill_diagonal_(float('-inf'))
            loss = F.cross_entropy(logits, pos_idx)
            return loss
        
        # Symmetric NT-Xent with queue embeddings appended only as negatives.
        z = torch.cat([z_i, z_j], dim=0)
        global_queue = F.normalize(global_queue, dim=1)
        local_logits = z @ z.T / self.T
        positive_indices = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        local_logits.fill_diagonal_(float('-inf'))

        queue_logits = z @ global_queue.T / self.T
        if self.queue_weight <= 0:
            raise ValueError("queue_weight must be positive")
        if self.queue_weight != 1.0:
            # Apply queue weighting in probability space.
            queue_logits = queue_logits + math.log(self.queue_weight)

        logits = torch.cat([local_logits, queue_logits], dim=1)

        if self.beta > 0:
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            row_indices = torch.arange(2 * B, device=z.device)
            negative_mask[row_indices, row_indices] = False
            negative_mask[row_indices, positive_indices] = False
            negative_logits = logits[negative_mask].view(2 * B, -1)
            count = negative_logits.size(1)
            log_normalizer = (
                torch.logsumexp(self.beta * negative_logits, 1, keepdim=True)
                - math.log(count)
            )
            reweighted = (self.beta + 1) * negative_logits - log_normalizer
            logits[negative_mask] = reweighted.reshape(-1)

        return F.cross_entropy(logits, positive_indices)
