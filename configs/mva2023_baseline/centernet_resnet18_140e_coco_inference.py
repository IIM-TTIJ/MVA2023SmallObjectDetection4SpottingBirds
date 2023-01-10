_base_ = './centernet_resnet18_140e_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json',
        img_prefix=data_root + 'mva2023_sod4bird_pub_test/images/',
    ) 
)

