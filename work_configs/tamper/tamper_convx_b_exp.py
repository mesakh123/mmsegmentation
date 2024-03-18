norm_cfg = dict(type="SyncBN", requires_grad=True)
custom_imports = dict(imports="mmpretrain.models", allow_failed_imports=False)
checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth"  # noqa
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=(512, 512),
)
num_classes = 2
metainfo = dict(
    classes=(
        "normal",
        "forged",
    ),
    palette=[[0, 0, 0], [255, 255, 255]],
)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="base",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
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
        out_channels=num_classes,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0, 1],
            ),
            dict(
                type="LovaszLoss",
                loss_weight=1,
                per_image=True,
                class_weight=[0.1, 1],
            ),
        ],
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000),
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
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0, 1],
            ),
            dict(
                type="LovaszLoss",
                loss_weight=1,
                per_image=True,
                class_weight=[0.1, 1],
            ),
        ],
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)


# dataset settings
dataset_type = "TamperDataset"
data_root = "data/"
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize", scale=(1024, 512), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomCopyMove", prob=0.7, mix_prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path="train/images", seg_map_path="train/masks"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path="val/images", seg_map_path="val/masks"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator


## Default

default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

tta_model = dict(type="SegTTAModel")


optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={"decay_rate": 0.9, "decay_type": "stage_wise", "num_layers": 12},
    constructor="LearningRateDecayOptimizerConstructor",
    loss_scale="dynamic",
)

max_iters = 100
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=1, begin=0, end=max_iters, by_epoch=False)
]
# training schedule for 40k


nx = 6
total_epochs = int(round(12 * nx))
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=total_epochs, save_optimizer=False)


train_cfg = dict(
    type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters / 10
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=4000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)


work_dir = f"./work_dirs/tamper/convx_b_exp_{nx}x_lova1_aug1.1_dec1_cpmv"
