echo "-- Processing test set..."
python data_argo/run_preprocess.py --mode test \
   --data_dir ~/Desktop/JupyterNotebook/SIMPL/Dataset/argodataset/argoverse/test_obs/data/ \
   --save_dir data_argo/features/ \
   --small