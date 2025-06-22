from simpl.av1_bezier_loss import BezierLossFunc
from utils.bezier_evaluator import TrajPredictionEvaluator
import torch

class AdvCfg:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train = {
            'seed': 1234,
            'device': 'cuda',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'epochs': 50,
            'batch_size': 16,
            'log_interval': 10,
            'save_interval': 5,
            'g_pred_len': 30,
            'cls_th': 5.0,
            'cls_ignore': 10.0,
            'mgn': 0.1,
            'cls_coef': 1.0,
            'reg_coef': 1.0,
            'data_aug': False,
            'g_num_modes': 6,
            'miss_thres': 2.0,
            'data_ver': 'av1',
        }

        self.loss_fn = BezierLossFunc(self.train, self.device)
        self.evaluator = TrajPredictionEvaluator(self.train)
