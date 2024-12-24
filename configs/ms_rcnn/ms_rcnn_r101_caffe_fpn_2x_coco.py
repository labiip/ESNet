_base_ = './ms_rcnn_r101_caffe_fpn_1x_coco.py'
# learning policy
#lr_config = dict(step=[16, 22])
lr_config = dict(step=[33, 44])
#runner = dict(type='EpochBasedRunner', max_epochs=24)
runner = dict(type='EpochBasedRunner', max_epochs=50)
