CUDA_VISIBLE_DEVICES=0 python train.py \
  --features_dir data_argo_dgfnet/features/ \
  --train_batch_size 16 \
  --val_batch_size 16 \
  --val_interval 2 \
  --train_epoches 30 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path config.DGFNet_cfg
