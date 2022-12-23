_base_ = './centernet_resnet18_140e_coco.py'
data_root = 'data/drone/'

data = dict(
    train=dict(
        ann_file=data_root + 'annotation/split_train_mini_coco.json',
    ),
    val=dict(
        ann_file=data_root + 'annotation/split_val_mini_coco.json',
    ),
    test=dict(
        ann_file=data_root + 'annotation/split_val_mini_coco.json',
    )
)

