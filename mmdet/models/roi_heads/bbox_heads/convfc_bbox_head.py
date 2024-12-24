import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead

import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from torch import functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.core import bbox_overlaps
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs'),
                        dict(name='iou_fc1'),
                        dict(name='reg_cls_fc'),
                        dict(name='fc_iou')
                    ])
            ]
        self.iou_fc1 = nn.Linear(12544,1024)
        self.reg_cls_fc= nn.Linear(12544, 1024)

        self.fc_iou = nn.Linear(1024, 1)
        
        self.cls_reg_conv_num = 4
        self.iou_conv = ConvModule(in_channels=512,out_channels=256,kernel_size=1,padding=0)

        self.cls_convs_ = nn.ModuleList()
        self.iou_convs_ = nn.ModuleList()
        for num in range(self.cls_reg_conv_num):
            self.cls_convs_.append(ConvModule(in_channels=256,out_channels=256,kernel_size=3,padding=1))
            self.iou_convs_.append(ConvModule(in_channels=256,out_channels=256,kernel_size=3,padding=1))

        self.iou_loss = build_loss(dict(
                     type='SmoothL1Loss',
                        beta=1.0,
                     loss_weight=1.0))
        self.cls_loss = build_loss(dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0))
        self.reg_loss = build_loss(dict(
                     type='SmoothL1Loss',
                    beta=1.0,
                     loss_weight=1.0))
        # self.reg_loss = build_loss(dict(type='BoundedIoULoss', loss_weight=1.0))

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
       if self.num_shared_convs > 0:
           for conv in self.shared_convs:
               x = conv(x)

       if self.num_shared_fcs > 0:
           if self.with_avg_pool:
               x = self.avg_pool(x)

           x = x.flatten(1)

           for fc in self.shared_fcs:
               x = self.relu(fc(x))
       # separate branches
       x_cls = x
       x_reg = x

       for conv in self.cls_convs:
           x_cls = conv(x_cls)
       if x_cls.dim() > 2:
           if self.with_avg_pool:
               x_cls = self.avg_pool(x_cls)
           x_cls = x_cls.flatten(1)
       for fc in self.cls_fcs:
           x_cls = self.relu(fc(x_cls))

       for conv in self.reg_convs:
           x_reg = conv(x_reg)
       if x_reg.dim() > 2:
           if self.with_avg_pool:
               x_reg = self.avg_pool(x_reg)
           x_reg = x_reg.flatten(1)
       for fc in self.reg_fcs:
           x_reg = self.relu(fc(x_reg))

       cls_score = self.fc_cls(x_cls) if self.with_cls else None
       bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
       return cls_score, bbox_pred
        # reg_cls_feat=x
        # iou_feat = x
        # for conv in self.cls_convs_:
        #     reg_cls_feat = conv(reg_cls_feat)
        #
        # iou_feat = torch.cat((iou_feat ,reg_cls_feat),dim=1)
        # iou_feat = self.iou_conv(iou_feat)
        #
        # for conv in self.iou_convs_:
        #     iou_feat = conv(iou_feat)
        #
        # reg_cls_fc = torch.flatten(reg_cls_feat,start_dim=1)
        #
        # reg_cls_fc = self.relu(self.reg_cls_fc(reg_cls_fc))
        # cls_score = self.fc_cls(reg_cls_fc)
        # bbox_pred = self.fc_reg(reg_cls_fc)
        #
        # iou_fc = torch.flatten(iou_feat, start_dim=1)
        # iou_fc =self.relu(self.iou_fc1(iou_fc))
        #
        # iou_pred = self.fc_iou(iou_fc).sigmoid()
        # return cls_score, bbox_pred,iou_pred
    def loss_(self,
             cls_score,
             bbox_pred,
             iou_pred,
             rois,

             labels,
             label_weights,

             bbox_targets,
             bbox_weights,

             iou_target,
             iou_weights,
             reduction_override=None):
        losses = dict()
        '''
        reset the bbox weights bbox_pred & bbox_targets calculate IoU,get mIoU follow the original lable or box weight
        IoU = bbox_overlaps()
        
        '''

        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        losses['loss_cls'] = self.cls_loss(
            cls_score,
            labels,
            label_weights,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        losses['acc'] = accuracy(cls_score, labels)

        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        if pos_inds.any():
            if self.reg_decoded_bbox:
                bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                       labels[pos_inds.type(torch.bool)]]

                pos_iou_pred = iou_pred.view(
                    bbox_pred.size(0),
                    1)[pos_inds.type(torch.bool),labels[pos_inds.type(torch.bool)]]
     
            losses['loss_bbox'] = self.reg_loss(
                pos_bbox_pred,  
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses['loss_iou'] = self.iou_loss(pos_iou_pred,
                                               iou_target[pos_inds.type(torch.bool)],
                                               iou_weights[pos_inds.type(torch.bool)],
                                               avg_factor=pos_bbox_pred.size(0),
                                               reduction_override=reduction_override)

        return losses
    def get_targets_(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]  
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights,iou_targets,iou_weights = multi_apply(
            self._get_target_single_,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        # box_targets == box_assign_GTBOX?

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)  
            bbox_targets = torch.cat(bbox_targets, 0)  
            bbox_weights = torch.cat(bbox_weights, 0)  
            iou_targets = torch.cat(iou_targets,0)
            iou_weights = torch.cat(iou_weights,0)
        return labels,label_weights,bbox_targets,bbox_weights,iou_targets,iou_weights


    def _get_target_single_(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        ## repulsion loss
        # 1> repul gt,,,decoeded bbox -- gtbox calculate iou(as_align=False) select secondary max iou GT loss 'iof', only positive sample
        # 2> repul box,,decoeded bbox -- bbox to bbox iou as_align=False) select all  positive bbox(iou>0), only positive sample

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)

        iou_targets = pos_bboxes.new_zeros(num_samples)
        iou_weights = pos_bboxes.new_zeros(num_samples)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:  # True
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets  
            bbox_weights[:num_pos, :] = 1
            '''
            ERROR!!!!!,
            pos_bboxes, boxes style is xywh,
            pos_gt_bboxes, boxes style is xywh,
            FUNC(bbox_overlaps) input style is xyxy,
            SO, THERE IS ERROR!!!!!!
            '''
            pos_iou_targets = bbox_overlaps(pos_bboxes, pos_gt_bboxes, mode='iou', is_aligned=True)

            iou_targets[:num_pos] = pos_iou_targets
            iou_weights[:num_pos] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, iou_targets, iou_weights

    def get_targets_with_iou(self,sampling_results,
                    gt_bboxes,
                    gt_labels,
                    pre_box_decode,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
                the sampling_results.

                Almost the same as the implementation in bbox_head, we passed
                additional parameters pos_inds_list and neg_inds_list to
                `_get_target_single` function.

                Args:
                    sampling_results (List[obj:SamplingResults]): Assign results of
                        all images in a batch after sampling.
                    gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                        each tensor has shape (num_gt, 4),  the last dimension 4
                        represents [tl_x, tl_y, br_x, br_y].
                    gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                        each tensor has shape (num_gt,).
                    rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
                    concat (bool): Whether to concatenate the results of all
                        the images in a single batch.

                Returns:
                    Tuple[Tensor]: Ground truth for proposals in a single image.
                    Containing the following list of Tensors:

                        - labels (list[Tensor],Tensor): Gt_labels for all
                          proposals in a batch, each tensor in list has
                          shape (num_proposals,) when `concat=False`, otherwise
                          just a single tensor has shape (num_all_proposals,).
                        - label_weights (list[Tensor]): Labels_weights for
                          all proposals in a batch, each tensor in list has
                          shape (num_proposals,) when `concat=False`, otherwise
                          just a single tensor has shape (num_all_proposals,).
                        - bbox_targets (list[Tensor],Tensor): Regression target
                          for all proposals in a batch, each tensor in list
                          has shape (num_proposals, 4) when `concat=False`,
                          otherwise just a single tensor has shape
                          (num_all_proposals, 4), the last dimension 4 represents
                          [tl_x, tl_y, br_x, br_y].
                        - bbox_weights (list[tensor],Tensor): Regression weights for
                          all proposals in a batch, each tensor in list has shape
                          (num_proposals, 4) when `concat=False`, otherwise just a
                          single tensor has shape (num_all_proposals, 4).
                """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self.get_target_with_iou_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pre_box_decode,
            pos_assigned_gt_inds_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def get_target_with_iou_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,pre_box_decode,
                           pos_assigned_gt_inds, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            pre_box_decode(Temsor): decoder predict box
                represents [tl_x, tl_y, br_x, br_y].
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # Need pos_box corresponding to gt_ids     |pos_assigned_gt_inds|
        pos_pre_box = pre_box_decode[:num_pos]
        pre_gt_ious = bbox_overlaps(pos_pre_box, pos_gt_bboxes, is_aligned=True)

        gt_ids = torch.unique(pos_assigned_gt_inds)
        pos_iou_weight = pos_bboxes.new_full((num_pos, ),
                                     1,
                                     dtype=torch.float32)
        iou_weight = pos_bboxes.new_full((num_samples, ),
                                     0,
                                     dtype=torch.float32)

        for i in gt_ids:
            mask = torch.where(pos_assigned_gt_inds==i, )[0]
            i_item = pre_gt_ious[mask]
            i_item = i_item / torch.max(i_item)
            pos_iou_weight[mask] = i_item
        iou_weight[:num_pos] = pos_iou_weight

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return labels, label_weights, bbox_targets, bbox_weights

    def xywh_to_xyxy(self, xywh):
        ct_x, ct_y, w, h = xywh[:,0], xywh[:,1], xywh[:,2], xywh[:,3]
        lb_x = ct_x - w / 2
        lb_y = ct_y - h / 2
        rt_x = ct_x + w / 2
        rt_y = ct_y + h / 2
        xyxy = torch.cat([lb_x, lb_y, rt_x, rt_y], dim=0)
        return xyxy

@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
