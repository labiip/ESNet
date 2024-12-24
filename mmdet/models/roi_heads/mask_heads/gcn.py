import torch
from torch import nn as nn
import numpy as np
import torch

import math
from torch.nn import functional as F
from torch.autograd import Variable
from mmcv.cnn import Conv2d
__all__ = ['gcn']
class gcn(nn.Module):
    def __init__(self,in_channels):
        super(gcn, self).__init__()
        self.in_channels = in_channels
        self.cov = Conv2d(self.in_channels,self.in_channels,kernel_size=3,padding=1,stride=1)
        self.query_transform_bound_bo = Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                                               bias=False)
        self.key_transform_bound_bo = Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1,
                                               padding=0,
                                               bias=False)
        self.value_transform_bound_bo = Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1,
                                               padding=0,
                                               bias=False)
        self.query_transform_bound_bo = Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1,
                                               padding=0,
                                               bias=False)
        self.output_transform_bound_bo = Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1,
                                               padding=0,
                                               bias=False)
        self.scale = 1.0 / (self.in_channels ** 0.5)
        self.blocker_bound_bo = nn.BatchNorm2d(self.in_channels, eps=1e-04)


    def forward(self,x):
        B, C, H, W = x.size()

        x_query_bound_bo = self.query_transform_bound_bo(x).view(B, C, -1)
        x_query_bound_bo = torch.transpose(x_query_bound_bo, 1, 2)

        x_key_bound_bo = self.key_transform_bound_bo(x).view(B, C, -1)

        x_value_bound_bo = self.value_transform_bound_bo(x).view(B, C, -1)
        x_value_bound_bo = torch.transpose(x_value_bound_bo, 1, 2)

        x_w_bound_bo = torch.matmul(x_query_bound_bo, x_key_bound_bo) * self.scale
        x_w_bound_bo = F.softmax(x_w_bound_bo, dim=-1)
        x_relation_bound_bo = torch.matmul(x_w_bound_bo, x_value_bound_bo)
        x_relation_bound_bo = torch.transpose(x_relation_bound_bo, 1, 2)
        x_relation_bound_bo = x_relation_bound_bo.view(B, C, H, W)

        x_relation_bound_bo = self.output_transform_bound_bo(x_relation_bound_bo)
        x_relation_bound_bo = self.blocker_bound_bo(x_relation_bound_bo)

        x = x + x_relation_bound_bo

        return x

if __name__=='__main__':
    model = gcn(in_channels=256)
    input = torch.randn(4,256,14,14)
    out = model(input)
    print(out.shape)

