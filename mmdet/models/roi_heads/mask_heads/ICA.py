# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import time
__all__ = ['IAM', 'SAM']

class IAM(nn.Module):
    def __init__(self,in_channels, reduction, use_wcs=False):
        super(IAM, self).__init__()
        '''
        # Module: Instance Attentio Mudle
        # Param in_channels: input channel number
        # Param reduction: channel reduction
        # Param use_wcs: whether use weight clipping strategy
        '''
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_wcs = use_wcs
        self.inter_channels = int(self.in_channels / self.reduction)

        self.reduction_conv = ConvModule(self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.value_conv = ConvModule(self.in_channels, self.in_channels, kernel_size=1)
        self.wcs = ConvModule(self.in_channels, out_channels=1, kernel_size=1)

    def forward(self, features):
        ins_features = self.reduction_conv(features)
        b, c, h, w = ins_features.shape
        similarity_mask = self.cal_similarity(ins_features, ins_features).sigmoid()
        # similarity_mask_fuse = similarity_mask# shape [b, hw, hw] and normalization


        val_features = self.value_conv(features) # [b,c,h,w]
        val_features_trnas = val_features.reshape(b, c * self.reduction, -1) # [b, c, hw]
        # if self.use_wcs:
        #     wcs_features = self.wcs(val_features) # shape [b,1,h,w]
        weight_mask = torch.matmul(val_features_trnas, similarity_mask).reshape(b, c*self.reduction ,h, w) # [b, c, hw]

        out = weight_mask + features
        return out, similarity_mask

    def cal_similarity(self, A, B):
        assert len(A.size()) == len(B.size()) == 4, 'Please transfer tensor to 4 dim'
        b, c, h, w = A.shape
        A = A.permute(0, 2, 3, 1).reshape(b, -1, c)
        B = B.permute(0, 2, 3, 1).reshape(b, -1, c).permute(0, 2, 1)
        A2 = torch.matmul(A, B)
        a_sq = A.pow(2).sum(axis=-1)
        b_sq = B.pow(2).sum(axis=-2)
        A_sq = a_sq[..., None].repeat(1, 1, h * w)
        B_sq = b_sq[:, None, ...].repeat(1, h * w, 1)
        D = A_sq + B_sq - 2 * A2
        D = torch.exp(-D)
        return D

class SAM(nn.Module):
    def __init__(self, in_channels, reduction, coord_size=14, use_wcs=True):
        super(SAM, self).__init__()
        self.in_channels = in_channels
        self.reduction =reduction
        self.inter_channels = int(in_channels / reduction)
        self.Q = ConvModule(self.in_channels, self.inter_channels, kernel_size=1)
        self.K = ConvModule(self.in_channels, self.inter_channels, kernel_size=1)
        self.V = ConvModule(self.in_channels, self.in_channels, kernel_size=1)
        self.size = coord_size
        self.coor_conv = ConvModule(in_channels=2, out_channels=self.inter_channels, kernel_size=1)

    def forward(self, features):
        features_Q = self.Q(features)
        features_K = self.K(features)
        features_V = self.V(features)

        b, c, h, w = features_Q.shape
        coor_conv = self.get_coord_conv(batch_size=b, spatial_size=h,device=features_Q.device)
        coor_conv = self.coor_conv(coor_conv)
        features_Q = features_Q + coor_conv
        features_K = features_K + coor_conv
        similarity_mask = self.cal_similarity(features_Q, features_K)
        # normal
        similarity_mask_all = similarity_mask.sum(dim=-1, keepdim=True) + 1e-14 # to prevent nan
        similarity_mask = torch.div(similarity_mask, similarity_mask_all)
        # similarity_mask = similarity_mask.softmax(dim=-1) #[b, hw, hw]


        features_V_trans = features_V.reshape(b, c*self.reduction, -1)  #[b, c, hw]
        weight_mask = torch.matmul(features_V_trans, similarity_mask).reshape(b, c * self.reduction, h, w)
        out = weight_mask + features
        return out

    def get_coord_conv(self,batch_size, spatial_size, device):
        A = torch.linspace(start=-1, end=1, steps=spatial_size)[None,...].repeat(spatial_size,1).to(device)
        A = A[None,...].repeat(batch_size, 1, 1, 1) # size [B,1,spatial_size,spatial_size]
        B = A.permute(0,1,3,2) # transpose
        D = torch.cat([A,B], dim=1)
        D.requires_grad = False
        return D

    def cal_similarity(self, A, B):
        assert len(A.size()) == len(B.size()) == 4, 'Please transfer tensor to 4 dim'
        b, c, h, w = A.shape
        A = A.permute(0, 2, 3, 1).reshape(b, -1, c)
        B = B.permute(0, 2, 3, 1).reshape(b, -1, c).permute(0, 2, 1)
        A2 = torch.matmul(A, B)
        a_sq = A.pow(2).sum(axis=-1)
        b_sq = B.pow(2).sum(axis=-2)
        A_sq = a_sq[..., None].repeat(1, 1, h * w)
        B_sq = b_sq[:, None, ...].repeat(1, h * w, 1)
        D = A_sq + B_sq - 2 * A2
        D = torch.exp(-D)
        return D


if __name__ == '__main__':

    '''Test'''
    # model1 = SAM(in_channels=256, reduction=4, coord_size=14)
    # model2 = IAM(in_channels=256, reduction=4,)
    # input = torch.rand(256, 256, 14, 14)
    # start = time.time()
    # out = model1(input)
    # end = time.time()
    # # Cpu 0.25s
    # # GPU 0.01s
    # print('in GPU, finished in ',end-start, 's')



















