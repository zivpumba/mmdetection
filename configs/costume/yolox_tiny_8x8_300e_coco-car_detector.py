_base_ = '../yolox/yolox_s_8x8_300e_coco.py'

log_config = dict(
    hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'Cars_Detector', 'entity': 'pumba', 'name': 'exp17'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=200,
         bbox_score_thr=0.3)]
)


# model settings

model = dict(
    random_size_range=(40, 40),
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.375,
        init_cfg = dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
            prefix='backbone')),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96, num_classes=1))

img_scale = (640, 640)  # height, width

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# train_dataset = dict(pipeline=train_pipeline)
classes = ('car_normal',)
dataset_type = 'CocoDataset'
img_prefix='/home/ubuntu/mmdetection/data/image_set'

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/mmdetection/data/train.json',
        img_prefix=img_prefix,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/mmdetection/data/valid.json',
        img_prefix=img_prefix,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/mmdetection/data/valid.json',
        img_prefix=img_prefix,
        pipeline=test_pipeline))

# data = dict(
#     train=dict(
#         img_prefix='/home/ubuntu/mmdetection/data/image_set',
#         classes=classes,
#         ann_file='/home/ubuntu/mmdetection/data/train.json',
#         pipeline=train_pipeline),
#     val=dict(
#         img_prefix='/home/ubuntu/mmdetection/data/image_set',
#         classes=classes,
#         ann_file='/home/ubuntu/mmdetection/data/valid.json',
#         pipeline=train_pipeline),
#     test=dict(
#         img_prefix='/home/ubuntu/mmdetection/data/image_set',
#         classes=classes,
#         ann_file='/home/ubuntu/mmdetection/data/test.json',
#         pipeline=test_pipeline))
# data = dict(
#     train=train_dataset,
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
