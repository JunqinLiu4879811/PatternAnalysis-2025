# UNet – HipMRI 2D Prostate Segmentation (Normal)

Task: HipMRI 2D prostate segmentation  
Metric target: **Dice ≥ 0.75** (test set, prostate label)  
Data (Rangpur):  
- Root: `/home/groups/comp3710/HipMRI_Study_open`  
- Images: `semantic_MRs/`  Labels: `semantic_labels_only/`

How to run (to be updated):
- Train: `python train.py --data /home/groups/comp3710/HipMRI_Study_open --epochs 50 --out outputs/`
- Predict: `python predict.py --weights outputs/best.ckpt --image <one_slice> --out vis/`

