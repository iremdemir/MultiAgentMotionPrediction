from typing import Dict, List
import torch
import torch.nn as nn
from utils.bezier_utils import bezier_curve
from utils.utils import gpu, to_long

class BezierLossFunc(nn.Module):
    def __init__(self, config, device):
        super(BezierLossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out, data):
        """
        Args:
            out: Tuple/List with
                 out[0]: cls logits tensor list
                 out[1]: Bézier control points tensor list of shape [batch, n_modes, num_ctrl_points, 2]
            data: dict with GT trajectories and pad masks
        """
        cls = out[0]
        bezier_ctrl_pts = out[1]

        traj_pred = [bezier_curve(x, num_points=self.config['g_pred_len']) for x in bezier_ctrl_pts]  

        loss_out = self.pred_loss(
            (cls, traj_pred),
            gpu(data["TRAJS_FUT"], self.device),
            to_long(gpu(data["PAD_FUT"], self.device))
        )
        loss_out["loss"] = loss_out["cls_loss"] + loss_out["reg_loss"]
        return loss_out

    def pred_loss(self, out: Dict[str, List[torch.Tensor]], gt_preds: torch.Tensor, pad_flags: torch.Tensor):
        cls = out[0]
        traj_pred = out[1]

        cls = torch.cat([x for x in cls], 0)
        traj_pred = torch.cat([x for x in traj_pred], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in pad_flags], 0).bool()

        loss_out = dict()
        num_modes = self.config["g_num_modes"]
        num_preds = self.config["g_pred_len"]

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        traj_pred = traj_pred[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)

        dist = []
        for j in range(num_modes):
            dist.append(
                torch.sqrt(
                    ((traj_pred[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)

        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()

        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss

        traj_pred = traj_pred[row_idcs, min_idcs]
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(traj_pred[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss

        return loss_out
