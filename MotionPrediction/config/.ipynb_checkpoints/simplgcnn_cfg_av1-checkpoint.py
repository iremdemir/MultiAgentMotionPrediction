import os
import sys


class AdvCfg():
    def __init__(self):
        self.g_cfg = dict()
        self.g_cfg['g_num_modes'] = 6
        self.g_cfg['g_obs_len'] = 20
        self.g_cfg['g_pred_len'] = 30

    def get_dataset_cfg(self):
        data_cfg = dict()
        data_cfg['dataset'] = "simpl.av1_dataset:ArgoDataset"
        data_cfg.update(self.g_cfg)
        return data_cfg

    def get_net_cfg(self):
        net_cfg = dict()

        # Point to your simplgcnn module and class
        net_cfg["network"] = "simpl.simplgcnn:TrajectoryPredictor"

        # Input/output dims as per your simplgcnn
        net_cfg["input_dim"] = 2          # XY coordinate input dim
        net_cfg["hidden_dim"] = 64        # LSTM hidden dim
        net_cfg["gcn_dim"] = 64           # GCN output dim
        net_cfg["attn_heads"] = 4         # Number of MHSA heads
        net_cfg["bezier_degree"] = 3      # Bézier curve degree for decoder
        net_cfg["future_len"] = self.g_cfg['g_pred_len']

        net_cfg.update(self.g_cfg)
        return net_cfg

    def get_loss_cfg(self):
        loss_cfg = dict()
        loss_cfg["loss_fn"] = "simpl.av1_loss_fn:LossFunc"
        loss_cfg["cls_coef"] = 0.1
        loss_cfg["reg_coef"] = 0.9
        loss_cfg["mgn"] = 0.2
        loss_cfg["cls_th"] = 2.0
        loss_cfg["cls_ignore"] = 0.2
        loss_cfg.update(self.g_cfg)
        return loss_cfg

    def get_opt_cfg(self):
        opt_cfg = dict()
        opt_cfg['opt'] = 'adam'
        opt_cfg['weight_decay'] = 0.0
        opt_cfg['lr_scale_func'] = 'none'  # none/sqrt/linear

        # scheduler
        opt_cfg['scheduler'] = 'polyline'

        if opt_cfg['scheduler'] == 'cosine':
            opt_cfg['init_lr'] = 6e-4
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-5
        elif opt_cfg['scheduler'] == 'cosine_warmup':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['T_max'] = 50
            opt_cfg['eta_min'] = 1e-4
            opt_cfg['T_warmup'] = 5
        elif opt_cfg['scheduler'] == 'step':
            opt_cfg['init_lr'] = 1e-3
            opt_cfg['step_size'] = 40
            opt_cfg['gamma'] = 0.1
        elif opt_cfg['scheduler'] == 'polyline':
            opt_cfg['init_lr'] = 1e-4
            opt_cfg['milestones'] = [0, 5, 35, 40]
            opt_cfg['values'] = [1e-4, 1e-3, 1e-3, 1e-4]

        opt_cfg.update(self.g_cfg)
        return opt_cfg

    def get_eval_cfg(self):
        eval_cfg = dict()
        eval_cfg['evaluator'] = 'utils.evaluator:TrajPredictionEvaluator'
        eval_cfg['data_ver'] = 'av1'
        eval_cfg['miss_thres'] = 2.0
        eval_cfg.update(self.g_cfg)
        return eval_cfg
