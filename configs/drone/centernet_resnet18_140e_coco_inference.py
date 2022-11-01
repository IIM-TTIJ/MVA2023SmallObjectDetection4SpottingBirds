_base_ = './centernet_resnet18_140e_coco.py'
data_root = 'data/drone/'

data = dict(
    test=dict(
        ann_file=data_root + 'annotation/train_coco.json', 
    ) 
)

