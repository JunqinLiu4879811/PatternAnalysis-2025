Difficulty: **Easy** | Student ID: **48798112**

# UNet – HipMRI 2D Prostate Segmentation (Easy)

## Task
2D segmentation of the **prostate** (label id = **5**) from HipMRI slices (axial).  
All code lives under: `recognition/unet_hipmri2d_48798112/`  
Key files: `dataset.py`, `modules.py`, `train.py`, `predict.py`.

- ## Data (Rangpur)
- - Root: `/home/groups/comp3710/HipMRI_Study_open`
- - Images: `semantic_MRs/`
- - Labels: `semantic_labels_only/`
- - Split handling is inside `dataset.py` (train/val/test)
## Data
- Root: `C:\Users\0.0\Desktop\UQ Study2\COMP3710\report`
- Images: `semantic_MRs/`
- Labels: `semantic_labels_only/`
- Split: handled in `dataset.py` (70/15/15)

+ - Pairing rule: files like `Case_*_WeekK_LFOV.nii.gz` (image) are paired with
+   `Case_*_WeekK_SEMANTIC_LFOV.nii.gz` (label). 
+
+ **Cluster (Rangpur, optional)**
+ - Root: `/home/groups/comp3710/HipMRI_Study_open`
+ - Images: `semantic_MRs/`
+ - Labels: `semantic_labels_only/`


## Environment
- Platform: UQ Rangpur
- Conda env (course-provided): `conda activate comp3710`
- Python 3.10, PyTorch (already available in the course env)

## Training
A100 sbatch (6h), UNet(1-in,1-out), size=256, batch=16, BCE/SoftDice or BCE, early-save `best.ckpt`.

## Evaluation (test)
- **Test mean Hard Dice@0.5 = `0.78`**  (≥ 0.75, split=test, id=<ID>, ckpt=`outputs/best.ckpt`)
- Overlays: see `pred_out_test/overlays/` (3–6 examples inserted in the PDF).

## Repro
Commands as in the repo (§training/§inference). Data path, commit hash, and job id recorded below.
- Commit: `<HASH>`; Job id: `<324031>`; Data root: `/home/groups/comp3710/HipMRI_Study_open`.
