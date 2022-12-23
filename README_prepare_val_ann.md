# MVA2023 - Small Object Detection for Birds Challenge 


Change the following setting in `tti/val_to_coco.py`:



```
######################################################
ANNOTATION_DIR = os.path.join(LIB_ROOT_DIR, 'data/drone/annotation')
IMAGE_DIR = os.path.join(LIB_ROOT_DIR, 'data/drone/drone_dataset_new')
val_json_file = os.path.join(ANNOTATION_DIR, 'val.json')
#######################################

```
Here `val.json` is you old annotation file in [Akita-san's format](https://github.com/kakitamedia/drone_dataset). 


Then, 

```
pip install imagesize
python tti/val_to_coco.py

```