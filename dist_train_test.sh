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
