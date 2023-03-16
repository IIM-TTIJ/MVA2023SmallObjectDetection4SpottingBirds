# MVA2023 - Small Object Detection Challenge for Spotting Birds

<img src="http://www.mva-org.jp/mva2023/data/uploads/bird/samples-1.pdf.jpg" alt="mva-sod4b-sample" title="mva2023-ChallengeOnSmallObjectDetection4SpottingBirds-sample">


This repository includes the baseline code used in our [challenge](http://www.mva-org.jp/mva2023/challenge) .
It is built on MMDetection V2.24.1 (released on Apr 30, 2022, source code is downloaded from [here](https://github.com/open-mmlab/mmdetection/releases/tag/v2.24.1)).


### Important dates
|  Challenges Event  |  Date (always 23:59 PST)  |
| ---- | ---- |
| [Site online](http://www.mva-org.jp/mva2023/challenge) | 2022.12.8 |
| Release of training data and public test data | 2023.1.9 |
| Public test server online | 2023.1.10 |
| Public test server closed | 2023.4.14 |
| Fact sheets, code/executable submission deadline | 2023.4.21 |
| Paper submission deadline (only Research Category) | 2023.5.7 |
| Preliminary private test results release to the participants | 2023.6.15 |
| Camera ready due (only Research Category) | 2023.7.4 |

## News
[2023/02/10] Currently, some inconsistencies, such as resolution discrepancies, have been identified with respect to the annotation data in the drone2021 dataset. We are performing a comprehensive consistency check of the annotations. Please be patient for a while.  
[2023/01/17] **Fixed critical bugs**  
[2023/01/10] Released dataset   
[2022/12/29] Important dates, Links and References are available.  


## Dataset
**[Download Link](https://drive.google.com/drive/folders/1vTHiIelagbzPO795yhOdNUFh9u2XxZP-?usp=share_link)**  

Dataset Directory Structure
```
data
 ├ drone2021 (62.4GB)
 │  ├ images
 │  └ annotations
 ├ mva2023_sod4bird_train (9.4GB)
 │  ├ images
 │  └ annotations
 ├ mva2023_sod4bird_pub_test (8.7GB)
 │  ├ images
 │  └ annotations(including an empty annotation file)
 └ mva2023_sod4bird_private_test (4kB)
    ├ images(empty)
    └ annotations(empty)
```

The dataset is split into training / public test / private test sets. Participants can download the images and annotations of the training data, and the images of the public test data. The annotations of the public test data are hidden but the participants can obtain the evaluation score once their detection results are online submitted via the [**CodaLab web platform**](https://codalab.lisn.upsaclay.fr/competitions/9594). Participants cannot access the private test images. The private test will be conducted manually by the challenge organizer. The trainig data includes an extended version of the publicly available data (train1) published in [4] and newly released data for this competition (train2).
* Images and instances:
    * Train :
        * Train1(modified based on [[2]](https://github.com/kakitamedia/drone_dataset)) (drone2021): Consists of 47,260 images with 60,971 annotated bird instances.
        * Train2 (mva2023_sod4bird_train): Consists of 9,759 images with 29,037 annotated bird instances.
    * Public test (mva2023_sod4bird_pub_test) : Consists of 9,699 images.
    * Private test (mva2023_sod4bird_private_test): Consists of approximately 10,000 images.
* Data format :
    * Input : Image
    * Annotation : COCO format

After the challenge, the public test evaluation server will continue to run on CodaLab to promote further research on small object detection.

### About Datasets and Weights Available

Open source licensed programs, datasets, trained models and weights are allowed for use in this contest.
However, participants who use external datasets must indicate the source of the external datasets they used in the questionnaire (and paper) to be filled out at the time of final submission.

<strong>Not Available</strong>
* Programs and datasets that require a fee to acquire
* Usage of the program or dataset is permitted only for specific participants
* Those that infringe on the rights of third parties
* Handlabeling the public test dataset 
* Use of the public test dataset for learning

## Links
* [MVA Official Competition Site](http://www.mva-org.jp/mva2023/challenge)
* [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/9594)
* [Competiton train and public test Dataset Download Link](https://drive.google.com/drive/folders/1vTHiIelagbzPO795yhOdNUFh9u2XxZP-?usp=share_link)
  
  
## Organizers
  
### Technical Event Chair
Norimichi Ukita (Toyota Technological Institute)  
Yuki Kondo (TOYOTA MOTOR CORPORATION)  

### Staff
Kaikai Zhao (Toyota Technological Institute)  
Riku Miyata (Toyota Technological Institute)  
Kazutoshi Akita (Toyota Technological Institute)  
  
### Contributor
Takayuki Yamaguchi (Iwate Agricultural Research Center)  
  
### Adviser
Masatsugu Kidode (Nara Institute of Science and Technology)  
  
## Citation

```
@misc{sodbchallenge2023misc,
  title={{MVA2023 Small Object Detection Challenge for Spotting Birds}},
  author={Yuki Kondo and Norimichi Ukita and Takayuki Yamaguchi},
  note={\url{https://www.mva-org.jp/mva2023/challenge}},
  year={2023}}

Note: Not yet published and this title is tentative.
@inproceedings{sodbchallenge2023,
  title={{MVA2023 Small Object Detection Challenge for Spotting Birds}},
  author={Yuki Kondo and Norimichi Ukita and Takayuki Yamaguchi, [Winners]},
  booktitle={International Conference on Machine Vision and Applications},
  note={\url{https://www.mva-org.jp/mva2023/challenge}},
  year={2023}}

```


# About baseline code


This code was created with reference to MMDetection[3]. 
OpenMMLab provides a [Colab tutorial](https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb).
If you are not familiar with MMDetection, please read its [tutorial](https://mmdetection.readthedocs.io/en/stable/) for how to modify the baseline code (e.g., replacing the detectors, backbones, learning rate, dataset and other hyperparameters in config files).
Our modification from MMDetection are in configs/mva2023_baseline, mmdet/datasets/pipelines/transforms.py and mmdet/datasets/pipelines/loading.py.

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
mim install mmcv-full==1.6.0
```
The mmcv-full version higher than 1.6.0 is not compatible with this baseline code.

**Step 3.** Install our baseline code
```shell
git clone https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds.git
cd MVA2023SmallObjectDetection4SpottingBirds
pip install -v -e .
```


### Dataset 
**[Download Link](https://drive.google.com/drive/folders/1vTHiIelagbzPO795yhOdNUFh9u2XxZP-?usp=share_link)**  
Please put it at `data/` after you download and uncompress it.

Directory Structure
```
data
 ├ drone2021
 ├ mva2023_sod4bird_train
 ├ mva2023_sod4bird_pub_test
 └ mva2023_sod4bird_private_test
```



### Evaluation metrics
The evaluation in this repository is based on COCO mAP.  
The [COCO API[4]](https://github.com/cocodataset/cocoapi) is used to evaluate the detection results.
The evaluation of the [MVA2023 Challenge on Small Bird Detection competition](http://www.mva-org.jp/mva2023/challenge) is based on AP0.5.
We think that using the official COCO mAP is good for developing your method as metrics such as AP0.5, AP0.75, AP_samll 
are also reported, so we keep it here.


### Commands to run the code

We used CenterNet (backbone ResNet18) in the baseline code and obtained mAP 47.3 (training: drone2021/split_train_coco.json,  test: drone2021/split_val_coco.json).
With hard negative training for additional 20 epochs, mAP on drone2021/split_val_coco.json was improved to 51.0. 

We have prepared the commands for conducting the distributed training and test in [`dist_train_test.sh`](dist_train_test.sh).


```shell
#!/usr/bin/env bash
#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO 
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
GPU_NUM=2

###############################
# Step 1: normal training on data/drone2021
###############################
echo "###############################"
echo "Step 1: normal training on data/drone2021"
echo "###############################"
bash tools/dist_train.sh  configs/mva2023_baseline/centernet_resnet18_140e_coco.py $GPU_NUM


###############################
# Step 2: fine-tuning on data/mva2023_sod4bird_train
###############################
echo "###############################"
echo "Step 2: fine-tuning on data/mva2023_sod4bird_train"
echo "###############################"
bash tools/dist_train.sh  configs/mva2023_baseline/centernet_resnet18_140e_coco_finetune.py $GPU_NUM


###############################
# Step 3: Generate predictions on data/mva2023_sod4bird_train to select hard negatives examples
###############################
echo "###############################"
echo "Step 3: Generate predictions on data/mva2023_sod4bird_train to select hard negatives examples"
echo "###############################"
CONFIG=configs/mva2023_baseline/centernet_resnet18_140e_coco_sample_hard_negative.py
CHECKPOINT=work_dirs/centernet_resnet18_140e_coco_finetune/latest.pth
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
 hard_neg_example_tools/test_hard_neg_example.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/centernet_resnet18_140e_coco_finetune/train_coco_hard_negative.json \
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05
    
# The setting for 'hard-negative-config' is the default setting for generating the hard negative examples. Please feel free to modify it.
# --------------------------


###############################
# Step 4: Hard negative training  on data/mva2023_sod4bird_train
###############################
echo "###############################"
echo "Step 4: Hard negative training  on data/mva2023_sod4bird_train"
echo "###############################"
bash tools/dist_train.sh  configs/mva2023_baseline/centernet_resnet18_140e_coco_hard_negative_training.py $GPU_NUM


###############################
# Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json.
###############################
echo "###############################"
echo "Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json."
echo "###############################"
bash tools/dist_test.sh configs/mva2023_baseline/centernet_resnet18_140e_coco_inference.py work_dirs/centernet_resnet18_140e_coco_hard_negative_training/latest.pth \
2 --format-only --eval-options jsonfile_prefix=results

_time=`date +%Y%m%d%H%M`
mkdir -p submit/${_time}
SUBMIT_FILE=`echo ./submit/${_time}/results.json`
SUBMIT_ZIP_FILE=`echo ${SUBMIT_FILE//results.json/submit.zip}`
mv ./results.bbox.json $SUBMIT_FILE
zip $SUBMIT_ZIP_FILE $SUBMIT_FILE

```


To submit your detection result, first rename your resulting file to `results.json` so that
our Server can automatically evaluate your submission (other name is not acceptable), then compress your `results.json` to a zip file (any name is OK, e.g., submit.zip). 
The code in the last three lines of the above code automatically rename the resulting json file and generate the zip file for submission.

We have already uploaded a sample submission <a href="https://drive.google.com/drive/folders/1X7J2QQbUCYpXTrlNc-MIR9XS4kWMb0KN?usp=share_link">here</a> for your reference.


## References
[1] Hank Chen, Awesome Tiny Object Detection, https://github.com/kuanhungchen/awesome-tiny-object-detection  
[2] Fujii, Sanae, Kazutoshi Akita, and Norimichi Ukita. "Distant Bird Detection for Safe Drone Flight and Its Dataset." 2021 17th International Conference on Machine Vision and Applications (MVA). IEEE, 2021. https://github.com/kakitamedia/drone_dataset   
[3] Chen, Kai, et al. "MMDetection: Open mmlab detection toolbox and benchmark." arXiv preprint arXiv:1906.07155 (2019). https://github.com/open-mmlab/mmdetection  
[4] Piotr Dollar and Tsung-Yi Lin, COCO API, https://github.com/cocodataset/cocoapi  
