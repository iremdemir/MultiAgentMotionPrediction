CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --features_dir data_av2/features/ \
  --use_cuda \
  --mode val \
  --model_path saved_models/simpl_agent_av2_bezier_ckpt.tar \
  --adv_cfg_path config.DGFNet_cfg \
  --visualizer simpl.av2_visualizer:Visualizer \
  --seq_id -1