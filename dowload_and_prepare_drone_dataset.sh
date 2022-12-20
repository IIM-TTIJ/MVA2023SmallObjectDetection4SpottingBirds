#!/bin/bash


pip install gdown

# 62G
gdown https://drive.google.com/open?id=10_gyG5GQLNRX89SUuSG1xy8MSUlbNwzv&authuser=0
unzip drone_dataset.zip
mv drone_dataset data/drone/.

## annotation data (train.json, val.json) already included in this repository
#gdown gdown https://drive.google.com/u/0/uc?id=12ncamxah03UYFmCxN3JnazTzzmCajz-F
#unzip annotation.zip
#mv annotation data/drone/.

pip install imagesize
python to_coco.py


