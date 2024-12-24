# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS, build_loss

__all__ = ['SAM']

class SAM(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction,
                 conv_cfg=None,
                 norm_cfg=None,
                 ):
        super(SAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 1)
        self.Q = ConvModule(self.in_channels,
                            self.inter_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg) # 256 channel to xx channel
        self.K = ConvModule(self.in_channels,
                            self.inter_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg)
        self.V = ConvModule(self.in_channels,
                            self.inter_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg)
        self.conv_out = ConvModule(self.inter_channels,
                            self.in_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg)

    def forward(self, mask_feats, edge_feats):
        assert len(mask_feats.size()) == 4
        n = mask_feats.size(0)
        edge_weight = self.Q(edge_feats).view(n, self.inter_channels, -1) #[n, 64, 28x28]
        edge_weight = edge_weight.permute(0, 2, 1) #[n, 28x28, 64]
        mask_weight = self.K(mask_feats).view(n, self.inter_channels, -1)#[n, 64, 28x28]

        mask_value = self.V(mask_feats).view(n, self.inter_channels, -1)
        mask_value = mask_value.permute(0, 2, 1) #[n, 28x28, 64]

        weight = torch.matmul(edge_weight, mask_weight)
        weight = weight.softmax(dim=-1) #[n, 28x28, 28x28]

        y = torch.matmul(weight, mask_value) #[n, 28x28, 64]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *mask_feats.size()[2:])
        output = mask_feats + self.conv_out(y)
        return output

# if __name__ == '__main__':
#     # test
#     module = SAM(in_channels=256, reduction=4)
#     mask_input = torch.rand(2, 256, 28, 28)
#     edge_input = torch.rand(2, 256, 28, 28)
#     out = module(mask_input, edge_input)
#     print(out.shape)







