import torch
import torch.nn as nn


class BezierDecoder(nn.Module):
    """
    Decode Bézier control points and generate continuous trajectories.
    Args:
        input_dim: Feature dimension from encoder
        output_dim: Usually 2 (x, y)
        degree: Degree of Bézier curve (cubic=3)
        future_len: Number of future timesteps to predict
    """
    def __init__(self, input_dim, output_dim=2, degree=3, future_len=30):
        super().__init__()
        self.degree = degree
        self.future_len = future_len

        # Predict control points: (degree+1) points per trajectory
        self.control_point_layer = nn.Linear(input_dim, (degree + 1) * output_dim)

        # Precompute parameter t for Bézier curve sampling [0,1]
        self.register_buffer('t_values', torch.linspace(0, 1, steps=future_len).unsqueeze(1))  # [future_len, 1]

    def forward(self, x):
        """
        x: [B, N, input_dim]
        Returns:
            trajectories: [B, N, future_len, output_dim]
        """
        B, N, _ = x.shape
        ctrl_pts = self.control_point_layer(x)  # [B, N, (degree+1)*output_dim]
        ctrl_pts = ctrl_pts.view(B, N, self.degree + 1, -1)  # [B, N, degree+1, output_dim]

        # Compute Bernstein basis polynomials
        t = self.t_values.to(x.device)  # [future_len, 1]
        n = self.degree

        # Compute binomial coefficients
        binomial_coeffs = torch.tensor([self._binomial_coef(n, i) for i in range(n + 1)], device=x.device).float()  # [degree+1]

        # Bernstein basis: B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)
        t_pow = torch.pow(t, torch.arange(n + 1, device=x.device))  # [future_len, degree+1]
        one_minus_t_pow = torch.pow(1 - t, n - torch.arange(n + 1, device=x.device))  # [future_len, degree+1]

        basis = binomial_coeffs * t_pow * one_minus_t_pow  # [future_len, degree+1]

        # Multiply control points by basis and sum over control points dim
        # ctrl_pts: [B, N, degree+1, output_dim]
        # basis: [future_len, degree+1]
        # Output shape: [B, N, future_len, output_dim]

        traj = torch.einsum('li,bnij->bnlj', basis, ctrl_pts)  # [B, N, future_len, output_dim]
        return traj

    def _binomial_coef(self, n, k):
        from math import comb
        return comb(n, k)
