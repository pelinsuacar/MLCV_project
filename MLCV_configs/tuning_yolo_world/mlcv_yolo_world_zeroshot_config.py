_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 5
num_training_classes = 3
max_epochs = 50  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 50
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.0005
train_batch_size_per_gpu = 4
#load_from = 'pretrained_models/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth'
persistent_workers = False

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path='MLCV_yoloworld_finetuned_image_embeddings.npy',
             prompt_dim=text_channels,
             num_prompts=5, 
             freeze_prompt=True,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           #type='HuggingVisionBackbone',
                           #model_name ='google/vit-base-patch16-224'
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           with_text_model=False
                          ),
             
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=False,
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             #neck = None,
             bbox_head=dict(type='YOLOWorldHead',
                            head_module=dict(
                                type='YOLOWorldHeadModule',
                                freeze_all=False,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))



meta_info = dict(classes = ("camping car", "car", "pickup", "truck", "van"))
#meta_info = dict(classes = ("car", "van"))


# dataset settings
coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          metainfo = meta_info,
                          data_root='vehicle_detection_satellite_coco',
                          ann_file = 'train_ann.json',
                          data_prefix=dict(img='train_img'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        metainfo = meta_info,
                        data_root='vehicle_detection_satellite_coco',
                        ann_file='test_ann.json',
                        data_prefix=dict(img='test_img'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=_base_.test_pipeline)

coco_test_dataset = dict(type='YOLOv5CocoDataset',
                        metainfo = meta_info,
                        data_root='vehicle_detection_satellite_coco',
                        ann_file = 'final_annotations.json',
                        data_prefix=dict(img='final_images'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=_base_.test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)

test_dataloader = dict(dataset=coco_test_dataset)
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=2,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='SGD',
    lr=base_lr,
    momentum=0.937,
    nesterov=True,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu))

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 50, 20),
                     ann_file='vehicle_detection_satellite_coco/final_annotations.json',
                     metric='bbox',
                    classwise=True)

test_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 50, 20),
                     ann_file='vehicle_detection_satellite_coco/final_annotations.json',
                     metric='bbox',
                    classwise=True)
