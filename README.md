# SEUNET-PPIS
## Overview
This repository is the source code of SEUNET-PPIS, accompanied with datasets and features.

## Installation
```bash
git lfs clone git@github.com:SlamHT/SEUNET-PPIS.git
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate seunet
```

## Training

```bash
unzip raw.zip
python -u main.py --dataset=ppis --trainset=Train_335 --testset=Test_60 --epochs=40 --hidden_features=128 --radius=14 --Lambda=0.1 --K=35 --model=segnn --lmax_h=3 --lmax_attr=3 --layers=3 --high_layers=2 --subspace_type=weightbalanced --norm=instance --batch_size=1 --weight_decay=1e-8 --pool=avg
```

## File Structure

`main.py` instantiates the model based on input hyperparameters. 
`ppis/dataset.py` generates the datasets for training and evaluating.
`ppis/train.py` is the code for training and evaluating.
`models` contains the codes of network architectures of our model.
