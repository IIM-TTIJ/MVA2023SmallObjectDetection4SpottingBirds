# MVA2023 - Small Object Detection for Birds Challenge 

This repository includes the baseline code used in the challenge.
It is built on MMDetection V2.24.1 (released on Apr 30, 2022, source code is downloaded from [here](https://github.com/open-mmlab/mmdetection/releases/tag/v2.24.1).
For the latest version of [MMDetection](https://github.com/open-mmlab/mmdetection), 


### Installation

We follow the [MMDetection Installation Website](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)
to install mmdet.

**Step 0.** Create a conda environment and activate it.

```shell
conda create -n mva python=3.7
conda activate mva
```

**Step 1.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)


```shell
conda install pytorch torchvision -c pytorch
```
**Step 2.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```
**Step 3.** Install our baseline code
```shell
git clone https://github.com/cyoukaikai/MVA2023BirdDetection.git
cd MVA2023BirdDetection
pip install -v -e .
```

### Drone Dataset 
The training data can be downloaded from [here](https://github.com/kakitamedia/drone_dataset).
We prepare a script [`dowload_and_prepare_drone_dataset.sh'](dowload_and_prepare_drone_dataset) to automatically download the dataset, transfer the annotation to coco format, 
merge the three classes ('hawk', 'crow', 'wild bird') into one ('bird'). The script can also merge  
the train/val to a single train set.

To run the script, 
```
./dowload_and_prepare_drone_dataset.sh

```

Some output after running the script,
```
data/drone/annotation/val.json 5605 images 10070 boxes
out_path data/drone/annotation/val_coco.json
out_path data/drone/annotation/val_mini_coco.json
data/drone/annotation/train.json 42790 images 52036 boxes
out_path data/drone/annotation/train_coco.json
out_path data/drone/annotation/train_mini_coco.json
data/drone/annotation/train_val_coco_merged.json 48395 images 62106 boxes
```


### Evaluation metrics
The evaluation in this repository is based on COCO mAP.  
The [COCO API](https://github.com/cocodataset/cocoapi) is used to evaluate the detection results.
The evaluation of the MVA2023 Challenge on Small Bird Detection competition is based on AP0.5.
We think that using the official COCO mAP is good for developing your method as metrics such as AP0.5, AP0.75, AP_samll 
are also reported, so we keep it here.


### Commands to run the code

We used CenterNet (backbone ResNet18) in the baseline and obtained mAP 47.3.
With hard negative training for additional 20 epochs, mAP was improved to 51.0. 

We have prepared the commands for conducting the distributed training and test in [`dist_train_test.sh`](dist_train_test.sh).

A sample results for submission is `smaple_submission.zip`. 
To submit your detection result, first generate `results.json' (other name is not acceptable so that
our Server can automatically evaluate your submission), then compress your `results.json' to a zip file (any name is OK, e.g., 
submit.zip)



```shell
#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO 
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
GPU_NUM=2

###############################
# Step 1: normal training
###############################
bash tools/dist_train.sh  configs/drone/centernet_resnet18_140e_coco.py $GPU_NUM


###############################
# Step 2: Generate hard negative predictions
###############################
CONFIG=configs/drone/centernet_resnet18_140e_coco_inference.py
CHECKPOINT=work_dirs/centernet_resnet18_140e_coco/latest.pth
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPU_NUM \
    --master_port=$PORT \
 tti/test.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/centernet_resnet18_140e_coco/train_coco_hard_negative.json \
    --hard-negative-config 'num_max_det=10, pos_iou_thr=1e-5, score_thd=0.05, nms_thd=0.05'
    
# The setting for 'hard-negative-config' is the default setting for generating the hard negative examples. Please feel free to modify it.
# --------------------------


###############################
# Step 3: Hard negative training
###############################
bash tools/dist_train.sh  configs/drone/centernet_resnet18_140e_coco_hard_negative_training.py $GPU_NUM


###############################
# Step 4: To generate the prediction for submission
###############################
bash tools/dist_test.sh \
    configs/drone/centernet_resnet18_140e_coco_hard_negative_training.py \
    work_dirs/centernet_resnet18_140e_coco_hard_negative_training/latest.pth \
    2 \ # 2 gpus
    --format-only \
    --eval-options jsonfile_prefix=results



```




### References 

```
@misc{sodbchallenge2023misc,
title={Small Object Detection for Birds Challenge 2023},
author={Yuki Kondo, Norimichi Ukita},
howpublished={\url{https://www.mva-org.jp/mva2023/SODchallenge}},
year={2023}}

@inproceedings{sodbchallenge2023},
title={Small Object Detection for Birds Challenge 2023},
author={Yuki Kondo, Norimichi Ukita, [Winners]},
booktitle={International Conference on Machine Vision and Applications},
note={\url{https://www.mva-org.jp/mva2023/SODchallenge}},
year={2023}}

```
