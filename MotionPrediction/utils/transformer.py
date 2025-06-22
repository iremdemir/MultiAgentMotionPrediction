import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for agent feature refinement.
    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, N, embed_dim]
        attn_out, _ = self.mha(x, x, x)  # Self-attention: query=key=value=x
        return attn_out
