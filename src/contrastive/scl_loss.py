import torch


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al. 2020).
    
    Args:
        features: [2N, D] normalized feature vectors
        labels: [2N] class labels
        temperature: temperature parameter
        eps: small constant for numerical stability
        
    Returns:
        Scalar loss value
    """
    device = features.device
    labels = labels.contiguous().view(-1, 1)  # [2N, 1]
    
    # Create mask for positive pairs (same class)
    mask = torch.eq(labels, labels.T).float().to(device)  # [2N, 2N]
    
    # Compute similarity matrix
    logits = torch.div(
        torch.matmul(features, features.T),
        temperature
    )  # [2N, 2N]
    
    # Mask out self-comparisons
    diag = torch.eye(logits.size(0), dtype=torch.bool, device=device)
    logits_masked = logits.masked_fill(diag, -1e12)
    
    # Compute log probabilities
    exp_logits = torch.exp(logits_masked)
    exp_sum = exp_logits.sum(dim=1, keepdim=True) + eps
    log_prob = logits_masked - torch.log(exp_sum)
    
    # For each anchor, compute mean log_prob over positives
    mask_no_self = mask * (~diag).float()
    positives_per_row = mask_no_self.sum(dim=1)
    
    # Filter rows with no positives
    valid_rows = positives_per_row > 0
    if valid_rows.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    mean_log_prob_pos = (mask_no_self * log_prob).sum(dim=1) / (positives_per_row + 1e-12)
    
    # Return negative mean (loss to minimize)
    loss = -mean_log_prob_pos[valid_rows].mean()
    return loss
