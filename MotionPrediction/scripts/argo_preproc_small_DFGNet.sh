echo "-- Processing val set..."
python data_argo_dgfnet/run_preprocess.py --mode val \
  --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/val/data/ \
  --save_dir data_argo_dgfnet_small/features/ \
  --small
  # --debug --viz

echo "-- Processing train set..."
python data_argo_dgfnet/run_preprocess.py --mode train \
  --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/train/data/ \
  --save_dir data_argo_dgfnet_small/features/ \
  --small

echo "-- Processing test set..."
python data_argo_dgfnet/run_preprocess.py --mode test \
  --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/test_obs/data/ \
  --save_dir data_argo_dgfnet_small/features/ \
  --small