# optimizer
#optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    #warmup_iters=500,
    warmup_iters=900,
    warmup_ratio=0.001,
    #step=[16, 22]
    step=[30, 40]
    )
#runner = dict(type='EpochBasedRunner', max_epochs=24)
runner = dict(type='EpochBasedRunner', max_epochs=60)
