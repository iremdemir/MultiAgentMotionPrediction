from utils.bezier_utils import bezier_curve

def prepare_post_out_for_evaluator(model_output, config):
    bezier_ctrl_pts = model_output[1]
    traj_pred = [bezier_curve(x, num_points=config['g_pred_len']) for x in bezier_ctrl_pts]

    cls_logits = model_output[0]
    prob_pred = [torch.softmax(x, dim=1) for x in cls_logits]

    return {
        'traj_pred': torch.cat(traj_pred, 0),  # batch x modes x pred_len x 2
        'prob_pred': torch.cat(prob_pred, 0)   # batch x modes
    }
