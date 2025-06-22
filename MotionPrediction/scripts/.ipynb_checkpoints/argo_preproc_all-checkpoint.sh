echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/val/data/ \
  --save_dir data_argo/features/

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/train/data/ \
  --save_dir data_argo/features/

# echo "-- Processing test set..."
# python data_argo/run_preprocess.py --mode test \
#   --data_dir ~/data/dataset/argo_motion_forecasting/test_obs/data/ \
#   --save_dir data_argo/features/