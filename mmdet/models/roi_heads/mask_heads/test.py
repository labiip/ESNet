import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.roi_heads.mask_heads.sam import SAM

module = SAM(in_channels=256, reduction=4)

mask = torch.rand(2,256,28,28)
edge = torch.rand(2,256,28,28)

out = module(mask, edge)
print(out)