import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Basic Graph Convolutional Layer.
    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: [N, in_features]
        adj: [N, N] adjacency matrix
        """
        support = self.linear(x)  # [N, out_features]
        out = torch.matmul(adj, support)  # Graph conv aggregation
        return F.relu(out)
