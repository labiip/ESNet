import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.mask_heads.sam import SAM
from mmcv.cnn import Conv2d

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

class MSM(nn.Module):
    def __init__(self, in_channel=256, reduction=8):
        super(MSM, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.inter_channel = self.in_channel // self.reduction
        self.conv1 = ConvModule(self.in_channel, self.inter_channel, kernel_size=1)
        self.dilation1 = ConvModule(in_channels=self.inter_channel, out_channels=self.inter_channel,
                                    kernel_size=3, dilation=1,padding=1)
        self.dilation3 = ConvModule(self.inter_channel,self.inter_channel, kernel_size=3,
                                    dilation=2,padding=2)
        self.dilation5 = ConvModule(self.inter_channel, self.inter_channel, kernel_size=3,
                                    dilation=3, padding=3)
        self.down = ConvModule(self.inter_channel * 3, self.inter_channel,kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.dilation1(x)
        x2 = self.dilation3(x)
        x3 = self.dilation5(x)
        # return self.down(torch.cat([x1,x2,x3],dim=1))
        return x1+x2+x3
class MSGCN(nn.Module):
    def __init__(self,in_channel, reduction):
        super(MSGCN, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.Q = MSM(in_channel=self.in_channel, reduction=reduction)
        self.K = MSM(in_channel=self.in_channel, reduction=reduction)
        self.V = ConvModule(in_channels=256,out_channels=256//reduction, kernel_size=1)
        self.up = ConvModule(256//8, 256, kernel_size=1)
        self.alpha = nn.Parameter(torch.FloatTensor([0]),requires_grad=True)
    def forward(self,x):
        #print('alpha is',self.alpha.item())
        b,c,h,w = x.shape
        c = c // self.reduction
        Q = self.Q(x).reshape(b,c,-1)
        K = self.K(x).permute(0,2,3,1).reshape(b,-1,c)
        weight = torch.matmul(K,Q).softmax(dim=-1)# B hw hw
        V = self.V(x).reshape(b,c,-1)
        final = torch.matmul(V,weight).reshape(b,c,h,w)
        final = self.up(final)
        return final + x * self.alpha
@HEADS.register_module()
class ESMaskHead(BaseModule):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_edge=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(ESMaskHead, self).__init__(init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.loss_edge = build_loss(loss_edge)
        # -------------- Edge ---------------------
        kernel_ = torch.FloatTensor([[-1, -1, -1],
                                     [-1, 8, -1],
                                     [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)
        self.weight_ = nn.Parameter(data=kernel_, requires_grad=False)
        self.jiangwei_conv = ConvModule(in_channels=256,
                                        out_channels=256,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        self.SAM = SAM(in_channels=256, reduction=4)

        self.edge_num_convs = 3
        self.edge_convs = ModuleList()
        for i in range(self.edge_num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.edge_convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        upsample_edge_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_edge_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_edge_cfg_.update(
                in_channels=upsample_edge_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
        self.upsample_edge = build_upsample_layer(upsample_edge_cfg_)
        self.edge_out = Conv2d(self.conv_out_channels, num_classes, kernel_size=1, stride=1, padding=0)

        # -------------- Edge ---------------------

        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            # suppress warnings
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.MSGCN = MSGCN(256, 8)
        self.down = ConvModule(in_channels=256+256+1, out_channels=256, kernel_size=1)
        self.residual = nn.Sequential(ConvModule(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      ConvModule(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      )
        self.beta = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def init_weights(self):
        super(ESMaskHead, self).init_weights()
        for m in [self.upsample, self.upsample_edge,self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        for m in [self.edge_out,]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @auto_fp16()
    def forward(self, x):
        # print('beta is ',self.beta)
        for conv in self.convs:
            x = conv(x)
        x = self.MSGCN(x)

        if self.upsample is not None:
            mask_feats = self.upsample(x)
            edge_feats = self.upsample_edge(x)

            if self.upsample_method == 'deconv':
                mask_feats = self.relu(mask_feats)
                edge_feats = self.relu(edge_feats)

        edge_pred = self.edge_out(edge_feats)
        # fusion_features = self.residual(edge_feats) + edge_feats
        # fusion_features = self.beta * fusion_features
        fusion_features = mask_feats + self.beta * mask_feats
        mask_feats = torch.cat([mask_feats,fusion_features,edge_pred], dim=1)
        mask_feats = self.down(mask_feats)
        mask_pred = self.conv_logits(mask_feats)

        return mask_pred, edge_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets


    def get_edge_targets(self, gt_masks):
        gt_masks = gt_masks[:, None, :, :]  # ����ά��
        # gt_masks_down = F.interpolate(gt_masks, scale_factor=0.5)
        # # tic = time.time()
        # edge_gt = self.get_edge_gt(gt_masks_down)
        edge_gt = F.conv2d(gt_masks, self.weight_, padding=1).clamp(min=0)  #

        edge_gt_new = torch.zeros_like(edge_gt)
        # edge_gt_id = ((edge_gt > 99) & (edge_gt < 104)) #
        edge_gt_id = (edge_gt > 0.1)
        edge_gt_new[edge_gt_id] = 1

        pos = edge_gt_new.sum()
        total = edge_gt_new.numel()
        neg = total - pos
        weight = 1 / torch.log(1.1 + (pos / total))
        # weight = torch.log((total / pos) - 1)
        # print('weight is ', weight)


        # img1 = edge_gt_new.cpu().numpy()
        edge_gt_w_id = (edge_gt_new == 1)  # True or False

        # edge_gt_posw = edge_gt_w_id.cpu().numpy().sum(axis=(1,2,3))
        # edge_gt_nw = np.ones_like(edge_gt_posw)*(edge_gt_new.shape[2]*edge_gt_new.shape[3]) - edge_gt_posw
        # edge_gt_pnw = np.stack([edge_gt_posw, edge_gt_nw],1)
        # edge_gt_pnw = 1/np.log(edge_gt_pnw/(edge_gt_new.shape[2]*edge_gt_new.shape[3])+1.2)
        edge_gt_w = torch.ones_like(edge_gt)
        edge_gt_w[edge_gt_w_id] = weight  ###################
        # img5 = edge_gt_w.cpu().numpy()
        return edge_gt_new, edge_gt_w

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    @force_fp32(apply_to=('edge_pred',))
    def loss_e(self, edge_pred, edge_targets, edge_targets_w):

        loss_edge = self.loss_edge(edge_pred, edge_targets, edge_targets_w)
        # loss_edge = self.loss_edge(edge_pred, edge_targets)
        return loss_edge

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape (Tuple): original image height and width, shape (2,)
            scale_factor(float | Tensor): If ``rescale is True``, box
                coordinates are divided by this scale factor to fit
                ``ori_shape``.
            rescale (bool): If True, the resulting masks will be rescaled to
                ``ori_shape``.

        Returns:
            list[list]: encoded masks. The c-th item in the outer list
                corresponds to the c-th class. Given the c-th outer list, the
                i-th item in that inner list is the mask for the i-th box with
                class label c.

        Example:
            >>> import mmcv
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> det_bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> det_labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = torch.FloatTensor((1, 1))
            >>> rescale = False
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self.get_seg_masks(
            >>>     mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape,
            >>>     scale_factor, rescale
            >>> )
            >>> assert len(encoded_masks) == C
            >>> assert sum(list(map(len, encoded_masks))) == N
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels
        # No need to consider rescale and scale_factor while exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            img_h, img_w = ori_shape[:2]
        else:
            if rescale:
                img_h, img_w = ori_shape[:2]
            else:
                if isinstance(scale_factor, float):
                    img_h = np.round(ori_shape[0] * scale_factor).astype(
                        np.int32)
                    img_w = np.round(ori_shape[1] * scale_factor).astype(
                        np.int32)
                else:
                    w_scale, h_scale = scale_factor[0], scale_factor[1]
                    img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                        np.int32)
                    img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                        np.int32)
                scale_factor = 1.0

            if not isinstance(scale_factor, (float, torch.Tensor)):
                scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = bboxes / scale_factor

        # support exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            threshold = rcnn_test_cfg.mask_thr_binary
            if not self.class_agnostic:
                box_inds = torch.arange(mask_pred.shape[0])
                mask_pred = mask_pred[box_inds, labels][:, None]
            masks, _ = _do_paste_mask(
                mask_pred, bboxes, img_h, img_w, skip_empty=False)
            if threshold >= 0:
                masks = (masks >= threshold).to(dtype=torch.bool)
            else:
                # TensorRT backend does not have data type of uint8
                is_trt_backend = os.environ.get(
                    'ONNX_BACKEND') == 'MMCVTensorRT'
                target_dtype = torch.int32 if is_trt_backend else torch.uint8
                masks = (masks * 255).to(dtype=target_dtype)
            return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks according to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
