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

