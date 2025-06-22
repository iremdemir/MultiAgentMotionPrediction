import torch


def build_adjacency_matrix(agent_positions, eps=1e-6):
    """
    Build adjacency matrix using displacement-based Gaussian weights.
    Args:
        agent_positions: [B, N, 2]
        eps: small value to avoid div by zero
    Returns:
        adjacency: [B, N, N]
    """
    B, N, _ = agent_positions.shape
    adjacency = torch.zeros(B, N, N, device=agent_positions.device)

    for b in range(B):
        pos = agent_positions[b]  # [N, 2]
        dist = torch.cdist(pos, pos, p=2)  # [N, N] Euclidean distance matrix
        sigma = torch.mean(dist) + eps
        weights = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        adjacency[b] = weights

    # Normalize adjacency row-wise (D^-1 A)
    row_sum = adjacency.sum(dim=2, keepdim=True) + eps
    adjacency = adjacency / row_sum

    return adjacency
