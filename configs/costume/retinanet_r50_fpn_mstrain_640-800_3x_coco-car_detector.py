_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../common/mstrain_3x_coco.py'
]



log_config = dict(
    hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project': 'Cars_Detector', 'entity': 'pumba', 'name': 'exp23'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=20,
         bbox_score_thr=0.6)]
)
model = dict(
    backbone=dict(
        frozen_stages=1,
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_mstrain_3x_coco/retinanet_r50_fpn_mstrain_3x_coco_20210718_220633-88476508.pth', 
            prefix='backbone')))
model = dict(bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='Resize', img_scale=(320, 240), keep_ratio=True),
    dict(type='RandomErasing', erase_prob=0.5, min_area_ratio=0.01, max_area_ratio=0.1),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='RandomShift', shift_ratio=0.2, max_shift_px=32, filter_thr_px=1),
    dict(type='RandomCrop', crop_size=(50,50), crop_type='absolute', allow_negative_crop=False, recompute_bbox=False, bbox_clip_border=True),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Expand', mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4), seg_ignore_label=None, prob=0.7),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3, bbox_clip_border=True),
    dict(type='RandomAffine', 
         max_rotate_degree=45.0,
         max_translate_ratio=0.1,
         scaling_ratio_range=(0.5, 1.5),
         max_shear_degree=2.0,
         border=(0, 0),
         border_val=(114, 114, 114),
         min_bbox_size=5,
         min_area_ratio=0.2,
         max_aspect_ratio=20,
         bbox_clip_border=True,
         skip_filter=True)]


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)


dataset_type = 'CocoDataset'
classes = ('car_normal',)

images_loc = '/home/ubuntu/mmdetection/data/image_set'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file='/home/ubuntu/mmdetection/data/train.json',
            classes=classes,
            img_prefix=images_loc)),
    val=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/mmdetection/data/valid.json',
        classes=classes,
        img_prefix=images_loc),
    test=dict(
        type=dataset_type,
        ann_file='/home/ubuntu/mmdetection/data/test.json',
        classes=classes,
        img_prefix=images_loc))

runner = dict(type='EpochBasedRunner', max_epochs=20)



# SOLVER = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0001,
#     early_stopping=True,
#     patience=5,
#     checkpoint_config=dict(interval=1)
# 