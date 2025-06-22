import torch
import torch.nn as nn
from utils.gcn import GCNLayer
from utils.transformer import MultiHeadSelfAttention
from utils.utils_gcn import build_adjacency_matrix
from utils.bezier import BezierDecoder


class AgentFeatureEncoder(nn.Module):
    """
    Encodes input agent trajectories into feature embeddings using LSTM.
    Input shape: [B, N, T_obs, 2]
    Output shape: [B, N, hidden_dim]
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super(AgentFeatureEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        B, N, T, _ = x.size()
        x = x.view(B * N, T, -1)
        _, (hn, _) = self.lstm(x)
        return hn[-1].view(B, N, -1)


class InteractionGCNLayer(nn.Module):
    """
    Computes agent interactions using GCN.
    Input: agent_feats [B, N, F], agent_pos [B, N, 2]
    Output: [B, N, out_dim]
    """
    def __init__(self, in_dim, out_dim):
        super(InteractionGCNLayer, self).__init__()
        self.gcn = GCNLayer(in_dim, out_dim)

    def forward(self, agent_feats, agent_pos):
        B, N, _ = agent_feats.shape
        A = build_adjacency_matrix(agent_pos)  # [B, N, N]

        out = []
        for b in range(B):
            out.append(self.gcn(agent_feats[b], A[b]))
        return torch.stack(out, dim=0)


class TrajectoryPredictor(nn.Module):
    """
    Full trajectory prediction pipeline using:
    1. LSTM encoder
    2. GCN interaction modeling
    3. MHSA refinement
    4. Bézier decoder
    """
    def __init__(self, input_dim=2, hidden_dim=64, gcn_dim=64,
                 attn_heads=4, bezier_degree=3, future_len=30):
        super(TrajectoryPredictor, self).__init__()
        self.encoder = AgentFeatureEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.gcn_layer = InteractionGCNLayer(in_dim=hidden_dim, out_dim=gcn_dim)
        self.attn = MultiHeadSelfAttention(embed_dim=gcn_dim, num_heads=attn_heads)
        self.decoder = BezierDecoder(input_dim=gcn_dim, output_dim=2,
                                     degree=bezier_degree, future_len=future_len)

    def forward(self, agent_histories, agent_positions):
        # Step 1: Encode input trajectories
        agent_feats = self.encoder(agent_histories)  # [B, N, hidden_dim]

        # Step 2: Interaction via GCN
        gcn_out = self.gcn_layer(agent_feats, agent_positions)  # [B, N, gcn_dim]

        # Step 3: Multi-head self-attention
        attn_out = self.attn(gcn_out)  # [B, N, gcn_dim]

        # Step 4: Decode trajectories
        predicted_trajs = self.decoder(attn_out)  # [B, N, T_pred, 2]
        return predicted_trajs
