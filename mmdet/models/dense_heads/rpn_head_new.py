import copy
import warnings
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.ops import batched_nms
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class NewRPNHead(RPNTestMixin, AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = 3
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        super(NewRPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)


    def _init_layers(self):
        """Initialize layers of the head."""
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(3):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred



    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = super(NewRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                (height, width, 3).
            scale_factors (list[ndarray]): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        batch_size = cls_scores[0].shape[0]
        nms_pre_tensor = torch.tensor(
            cfg.nms_pre, device=cls_scores[0].device, dtype=torch.long)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(batch_size, -1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(-1)[..., 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4)
            anchors = mlvl_anchors[idx]
            anchors = anchors.expand_as(rpn_bbox_pred)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, rpn_bbox_pred.shape[1])
            if nms_pre > 0:
                _, topk_inds = scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    # Mind k<=3480 in TensorRT for TopK
                    transformed_inds = scores.shape[1] * batch_inds + topk_inds
                    scores = scores.reshape(-1, 1)[transformed_inds].reshape(
                        batch_size, -1)
                    rpn_bbox_pred = rpn_bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    anchors = anchors.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                else:
                    # sort is faster than topk
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:, :cfg.nms_pre]
                    scores = ranked_scores[:, :cfg.nms_pre]
                    batch_inds = torch.arange(batch_size).view(
                        -1, 1).expand_as(topk_inds)
                    rpn_bbox_pred = rpn_bbox_pred[batch_inds, topk_inds, :] # mei ceng tezheng tu tiaoxuan qian 1000 ge
                    anchors = anchors[batch_inds, topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((
                    batch_size,
                    scores.size(1),
                ),
                                idx,
                                dtype=torch.long))
        # here all box number is lower than 5000
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_rpn_bbox_pred = torch.cat(mlvl_bbox_preds, dim=1)
        batch_mlvl_proposals = self.bbox_coder.decode(
            batch_mlvl_anchors, batch_mlvl_rpn_bbox_pred, max_shape=img_shapes)
        batch_mlvl_ids = torch.cat(level_ids, dim=1)


        # deprecate arguments warning
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn(
                'In rpn_proposal or test_cfg, '
                'nms_thr has been moved to a dict named nms as '
                'iou_threshold, max_num has been renamed as max_per_img, '
                'name of original arguments and the way to specify '
                'iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You ' \
                    f'set max_num and ' \
                    f'max_per_img at the same time, but get {cfg.max_num} ' \
                    f'and {cfg.max_per_img} respectively' \
                    'Please delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                f' iou_threshold in nms and ' \
                f'nms_thr at the same time, but get' \
                f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                f' respectively. Please delete the nms_thr ' \
                f'which will be deprecated.'

        # Replace batched_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores.unsqueeze(2)
            score_threshold = cfg.nms.get('score_thr', 0.0)
            nms_pre = cfg.get('deploy_nms_pre', cfg.max_per_img)
            dets, _ = add_dummy_nms_for_onnx(batch_mlvl_proposals,
                                             batch_mlvl_scores,
                                             cfg.max_per_img,
                                             cfg.nms.iou_threshold,
                                             score_threshold, nms_pre,
                                             cfg.max_per_img)
            return dets

        result_list = []
        for (mlvl_proposals, mlvl_scores,
             mlvl_ids) in zip(batch_mlvl_proposals, batch_mlvl_scores,
                              batch_mlvl_ids): # bianli meige batch
            # Skip nonzero op while exporting to ONNX
            if cfg.min_bbox_size >= 0 and (not torch.onnx.is_in_onnx_export()):
                w = mlvl_proposals[:, 2] - mlvl_proposals[:, 0]
                h = mlvl_proposals[:, 3] - mlvl_proposals[:, 1]
                valid_ind = torch.nonzero(
                    (w > cfg.min_bbox_size)
                    & (h > cfg.min_bbox_size),
                    as_tuple=False).squeeze() #you xiao kuang torch,nonzero return not 0 index
                if valid_ind.sum().item() != len(mlvl_proposals):
                    mlvl_proposals = mlvl_proposals[valid_ind, :]
                    mlvl_scores = mlvl_scores[valid_ind]
                    mlvl_ids = mlvl_ids[valid_ind]

            dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
                                     cfg.nms)
            result_list.append(dets[:cfg.max_per_img]) # max box number is 1000
        # np.save('rpn_box.npy', result_list, allow_pickle=True)
        # print('has finished the box \n save....... save path is {}'.format(os.getcwd()))
        return result_list