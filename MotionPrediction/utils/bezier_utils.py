import torch
import math

def bezier_curve(control_points, num_points=30):
    """
    Compute sampled Bézier curve points from control points.

    Args:
        control_points: Tensor of shape [batch, n_modes, num_ctrl_points, 2]
        num_points: Number of points to sample along the curve

    Returns:
        Tensor of shape [batch, n_modes, num_points, 2]
    """
    device = control_points.device
    batch_size, n_modes, n_ctrl_points, _ = control_points.shape
    n = n_ctrl_points - 1  # degree

    t = torch.linspace(0, 1, num_points, device=device)  # [num_points]
    t = t.view(1, 1, num_points, 1)  # [1,1,num_points,1]

    def comb(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    
    binomial_coeffs = torch.tensor(
        [comb(n, i) for i in range(n_ctrl_points)],
        dtype=control_points.dtype,
        device=device
    ).view(1, 1, n_ctrl_points, 1)  # [1,1,num_ctrl_points,1]

    i = torch.arange(n_ctrl_points, device=device).view(1, 1, n_ctrl_points, 1)
    bernstein_poly = (
        binomial_coeffs
        * (t ** i)
        * ((1 - t) ** (n - i))
    )  # [1,1,num_ctrl_points,num_points]

    curve_points = torch.sum(control_points.unsqueeze(-1) * bernstein_poly, dim=2)  # [batch, n_modes, num_points, 2]

    return curve_points
