# optimizer
#optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) 8GPU/2img per GPU
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    #step=[8, 11]
    step=[33, 44])
#runner = dict(type='EpochBasedRunner', max_epochs=12)
runner = dict(type='EpochBasedRunner', max_epochs=50)
