# SEUNET-PPIS
## Overview
This repository is the source code of SEUNET-PPIS, accompanied with datasets and features.

## Installation
```bash
git lfs clone git@github.com:SlamHT/SEUNET-PPIS.git
```

## How to run

```bash
conda env create -f environment.yml
conda activate seunet
```

```bash
unzip raw.zip
python -u main.py --dataset=ppis --trainset=Train_335 --testset=Test_60 --epochs=40 --hidden_features=128 --radius=14 --Lambda=0.1 --K=35 --model=segnn --lmax_h=3 --lmax_attr=3 --layers=3 --high_layers=2 --subspace_type=weightbalanced --norm=instance --batch_size=1 --weight_decay=1e-8 --pool=avg
```

