# MVA2023 - Small Object Detection for Birds Challenge 

<img src="http://www.mva-org.jp/mva2023/data/uploads/bird/samples-1.pdf.jpg" alt="mva-sod4b-sample" title="mva2023-ChallengeOnSmallObjectDetection4Birds-sample">


This repository includes the baseline code used in our [challenge](http://www.mva-org.jp/mva2023/challenge) .
It is built on MMDetection V2.24.1 (released on Apr 30, 2022, source code is downloaded from [here](https://github.com/open-mmlab/mmdetection/releases/tag/v2.24.1).
For the latest version of [MMDetection](https://github.com/open-mmlab/mmdetection), 

### Important dates
|  Challenges Event  |  Date (always 23:59 PST)  |
| ---- | ---- |
| [Site online](http://www.mva-org.jp/mva2023/challenge) | 2022.12.8 |
| Release of training data and validation data | 2023.1.9 |
| Validation server online | 2023.1.10 |
| Validation server closed | 2023.4.14 |
| Fact sheets, code/executable submission deadline | 2023.4.21 |
| Paper submission deadline (only Research Category) | 2023.5.7 |
| Preliminary test results release to the participants | 2023.6.15 |
| Camera ready due (only Research Category) | 2023.7.4 |

## News
[2022/12/29] Important dates, Links and References are available.  
[2022/12/23] Our code is available.


### Installation

We follow the [MMDetection Installation Website](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)
to install mmdet.

**Step 0.** Create a conda environment and activate it.

```shell
conda create -n mva python=3.7
conda activate mva
```

**Step 1.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/)

We tested our code on `pytorch-1.10.2` and `pytorch-1.12.1`, please feel free to use other PyTorch versions.


**Step 2.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```
**Step 3.** Install our baseline code
```shell
git clone https://github.com/IIM-TTIJ/MVA2023BirdDetection.git
cd MVA2023BirdDetection
pip install -v -e .
```

### Drone Dataset 
To run this competition, we extended the [drone dataset](https://github.com/kakitamedia/drone_dataset). 
We collected new images and conducted annotation, the validation data and test data are from
the newly collected data. 
The training images can be downloaded from [here](https://drive.google.com/file/d/10_gyG5GQLNRX89SUuSG1xy8MSUlbNwzv/view).
Please put it at `data/drone/` after you download and uncompress it.


We provided the annotation in `data/drone/annotation` for the training images.

```
data/drone/annotation/val.json 5605 images 10070 boxes
out_path data/drone/annotation/split_val_coco.json
out_path data/drone/annotation/split_val_mini_coco.json
data/drone/annotation/train.json 42790 images 52036 boxes
out_path data/drone/annotation/split_train_coco.json
out_path data/drone/annotation/split_train_mini_coco.json
data/drone/annotation/merged_train.json 48395 images 62106 boxes
```
`merged_train.json` is the merged annotation of the original drone dataset, which will be used as the training 
set for this competition. 
`split_train_coco.json` and `split_val_coco.json` correspond to the original train/val splits in the original drone dataset. 
`split_train_mini_coco.json` and `split_val_mini_coco.json` can be used to debug your method.

The annotation we provided transfer the annotation in the original drone dataset to coco format, and merge the three classes ('hawk', 'crow', 'wild bird') to one ('bird'). 



### Evaluation metrics
The evaluation in this repository is based on COCO mAP.  
The [COCO API](https://github.com/cocodataset/cocoapi) is used to evaluate the detection results.
The evaluation of the [MVA2023 Challenge on Small Bird Detection competition](http://www.mva-org.jp/mva2023/challenge) is based on AP0.5.
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


## Links
* [MVA Official Competition Site](http://www.mva-org.jp/mva2023/challenge)
* CodaLab (to be announced)
* Dataset Download Link (to be announced)

## Citation

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

## References
* https://github.com/kuanhungchen/awesome-tiny-object-detection