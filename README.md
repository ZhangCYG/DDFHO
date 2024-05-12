# DDF-HO: Hand-Held Object Reconstruction via Conditional Directed Distance Field
Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, Xiangyang Ji 
in NeurIPS 2023

## Installation
```
conda env create -f docs/env.yaml
conda activate ddfho
```

## Data Preparation

### Folder Structure
```
data/
    cache/
    mesh_ddf/
        ddf_obj/
            obman/
            ho3d/
            mow/
    obman/
    ho3d/
    mow/

database/
    ShapeNetCore.v2/
    YCBmodels/

externals/
    mano/
```

### Download
To keep the training and testing split with IHOI (https://github.com/JudyYe/ihoi), we use their `cache` file (https://drive.google.com/drive/folders/1v6Pw6vrOGIg6HUEHMVhAQsn-JLBWSHWu?usp=sharing). Unzip it and put under `data/` folder.

`obman` is downloaded from https://hassony2.github.io/obman.

`ho3d` is downloaded from https://www.tugraz.at/index.php?id=40231 (we use HO3D(v2)).

`mow` is downloaded from https://zhec.github.io/rhoi/.

`externals/mano` contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`, get them from https://mano.is.tue.mpg.de/.

### DDF Preprocess
First prepare `ShapeNetCore.v2` for ObMan dataset and `YCBmodels` (We get the YCB models from https://rse-lab.cs.washington.edu/projects/posecnn/) for HO3D(v2) dataset.

Then, run
```
python preprocess/process_obman.py
python preprocess/process_ho3d.py
python preprocess/process_mow.py
```
and get processed DDF data under `processed_data`. You can make a soft link to `data/mesh_ddf/ddf_obj/`.

## Pretrained Models
We provide DDF-HO model pretrained on ObMan dataset (https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/zcyg22_mails_tsinghua_edu_cn/ErLUJGst6u9IlFUq4lS88XsB7eKExtCkhgLk2xtwSkuoBg?e=3LYO4F). Finetuning on HO3D and MOW datasets based on this model would be quick and convenient. 

## Train
```
python -m models.ddfho --config experiments/obman.yaml
python -m models.ddfho--config experiments/ho3d.yaml  --ckpt PATH_TO_OBMAN_MODEL
python -m models.ddfho--config experiments/mow.yaml  --ckpt PATH_TO_OBMAN_MODEL
```

## Test
```
python -m models.ddfho --config experiments/obman.yaml --eval --ckpt PATH_TO_OBMAN_MODEL
python -m models.ddfho --config experiments/ho3d.yaml --eval --ckpt PATH_TO_HO3D_MODEL
python -m models.ddfho --config experiments/mow.yaml --eval --ckpt PATH_TO_MOW_MODEL
```