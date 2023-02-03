_base_ = [
    '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person-bicycle-car.py',
]


log_config = dict(
    hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'Cars_Detector', 'entity': 'pumba', 'name': 'exp21'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=200,
         bbox_score_thr=0.3)]
)




# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        frozen_stages=1,
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth', 
            prefix='backbone')))
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

train_pipeline = [
    # dict(type='RandomErasing', erase_prob=0.5, min_area_ratio=0.01, max_area_ratio=0.1),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.2),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32, filter_thr_px=1),
    # dict(type='RandomCrop', crop_size=(50,50), crop_type='absolute', allow_negative_crop=False, recompute_bbox=False, bbox_clip_border=True),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Expand', mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4), seg_ignore_label=None, prob=0.7),
    # dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3, bbox_clip_border=True),
    dict(type='RandomAffine', 
         max_rotate_degree=45.0,
         max_translate_ratio=0.1,
         scaling_ratio_range=(0.5, 1.5),
         max_shear_degree=2.0,
         border=(0, 0),
         border_val=(114, 114, 114),
         min_bbox_size=2,
         min_area_ratio=0.2,
         max_aspect_ratio=20,
         bbox_clip_border=True,
         skip_filter=True)]

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('car_bush_normal',)

data = dict(
    train=dict(
        img_prefix='/home/ubuntu/mmdetection/data/image_set',
        classes=classes,
        ann_file='/home/ubuntu/mmdetection/data/train.json'),
    val=dict(
        img_prefix='/home/ubuntu/mmdetection/data/image_set',
        classes=classes,
        ann_file='/home/ubuntu/mmdetection/data/valid.json'),
    test=dict(
        img_prefix='/home/ubuntu/mmdetection/data/image_set',
        classes=classes,
        ann_file='/home/ubuntu/mmdetection/data/test.json'))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

 
runner = dict(type='EpochBasedRunner', max_epochs=20)
