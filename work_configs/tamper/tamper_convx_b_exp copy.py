from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop

num_classes = 2
default_scope = "mmseg"
size = 512
crop_size = 512
ratio = size / crop_size
# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth"  # noqa
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(size,size)
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="large",
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.75),
        #     dict(type='DiceLoss', loss_weight=0.25)
        #     ]
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="LovaszLoss", loss_weight=0.6, reduction="none"),
        ],
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="LovaszLoss", loss_weight=0.6, reduction="none"),
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
)

# dataset settings
dataset_type = "TamperDataset"
data_root = "data/"
classes = ["normal", "forged"]
palette = [[0, 0, 0], [255, 255, 255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

model["test_cfg"] = dict(mode="slide", stride=(size, size), crop_size=(size, size))
# crop_size = (256, 256)
albu_train_transforms = [
    dict(type="ColorJitter", p=0.5),
    # dict(type='GaussianBlur', p=0.5),
    # dict(type='JpegCompression', p=0.5, quality_lower = 75),
    # dict(type='Affine', rotate=5, shear=5, p=0.5),
    # dict(type='RandomResizedCrop', always_apply = True, height = crop_size, width = crop_size, scale = (0.9, 1.1), ratio = (1.0, 1.0), p=0.5),
    # dict(type='ToGray', p=0.5),
]
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomCrop", cat_max_ratio=0.75, crop_size=(crop_size, crop_size)),
    dict(type="RandomCopyMove", prob=0.5, mix_prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    # dict(type='RandomRotate90', prob=0.5),
    dict(type="Albu", transforms=albu_train_transforms),
    dict(type="Resize", scale=(size, size)),
    # dict(type='PhotoMetricDistortion'),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(size, size), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

train = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir="train/images",
    ann_dir="train/masks",
    img_suffix=".jpg",
    seg_map_suffix=".png",
    classes=classes,
    palette=palette,
    use_mosaic=False,
    mosaic_prob=0.5,
    pipeline=train_pipeline,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="train/images", seg_map_path="train/masks"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="val/images", seg_map_path="val/masks"),
        pipeline=test_pipeline,
    ),
)


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="CustomizedTextLoggerHook", by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None  # "./work_dirs/tamper/convx_t_8x/epoch_96.pth"
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True

nx = 6
total_epochs = int(round(12 * nx))
# optimizer

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
    constructor="LearningRateDecayOptimizerConstructor",
    loss_scale="dynamic",
)


# learning policy
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=total_epochs, save_optimizer=False)
fp16 = dict(loss_scale=512.0)

train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=4000)
val_cfg = dict(type=ValLoop)


val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"])

work_dir = f"./work_dirs/tamper/convx_b_exp_{nx}x_lova1_aug1.1_dec1_cpmv"
