import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
from mmcv import ConfigDict
from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.ops.roi_align import roi_align
import pdb
from mmdet.core.mask.structures import polygon_to_bitmap, BitmapMasks

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

class MDC(nn.Module):  # mutil dilation conv
    def __init__(self, channel_dim, out_channel, dilation=[1, 3, 5], use_pooling=False):
        super(MDC, self).__init__()
        self.fusion_conv = ConvModule(channel_dim, out_channel, kernel_size=1)
        for index, dila in enumerate(dilation):
            self.add_module(f'dila_con{index+1}',
                            ConvModule(out_channel, out_channel, kernel_size=3, padding=dila, dilation=dila))

        self.out_conv = ConvModule(out_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.fusion_conv(x) #256
        x1 = self.dila_con1(x)
        x2 = self.dila_con2(x)
        x3 = self.dila_con3(x)
        out = self.out_conv(x1 + x2 + x3)
        return out

# class MFeatureFusion(nn.module):
#     def __init__(self,):
#         super(MFeatureFusion, self).__init__()
#
#         self.low_feature_extractor = build_roi_extractor(dict(
#                 type='SingleRoIExtractor',
#                 roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
#                 out_channels=256,
#                 featmap_strides=[4, ]))
#         self.low_feature_conv = ConvModule()
#         self.mutil_feature_fusion = MDC(channel_dim=256)
#
#     def forward(self,segmantic_feat,roi_feature,egde_map,mask_map,rois):
#         low_feature_out = self.low_feature_extractor([segmantic_feat,],rois)

@HEADS.register_module()
class MFFHead(nn.Module):

    def __init__(self,
                 num_convs=3,
                 num_semantic_convs=2,
                 fusion_feature_conv=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 segmantic_in_channel=256,
                 segmantic_out_channel=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_wegiht=[0.25, 0.75, 1],
                 target_mask_size = [28,56],
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_edge=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_segmantic=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(MFFHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.num_semantic_convs = num_semantic_convs
        self.fusion_feature_conv = fusion_feature_conv
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size) #娌℃湁浣滅敤
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fusion_feature_convs = nn.ModuleList()
        for i in range(self.fusion_feature_conv):
            padding = (self.conv_kernel_size - 1) // 2
            self.fusion_feature_convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        self.convs = nn.ModuleList()
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

        self.semantic_convs = nn.ModuleList()
        for i in range(self.num_semantic_convs):
            padding = (self.conv_kernel_size - 1) // 2
            self.semantic_convs.append(
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
        #use deconv or bilinear?

        self.loss_mask = build_loss(loss_mask)
        self.loss_edge = build_loss(loss_edge)
        self.loss_segmantic = build_loss(loss_segmantic)


        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        self.loss_weight = loss_wegiht #assign different weight to different loss
        self.target_mask_size = target_mask_size

        self.mask28_out = ConvModule(in_channels,num_classes,kernel_size=1)
        self.edge28_out = ConvModule(in_channels, num_classes, kernel_size=1)
        self.edge_conv_28 = ConvModule(in_channels, 256, kernel_size=3, padding=1)

        self.semantic_out = ConvModule(in_channels,1,kernel_size=1)

        self.mask56_out = ConvModule(in_channels,num_classes,kernel_size=1)
        self.edge56_out = ConvModule(in_channels, num_classes, kernel_size=1)
        self.edge_conv_56 = ConvModule(in_channels, 256, kernel_size=3, padding=1)

        self.semantic_roi_extractor = build_roi_extractor(dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=28, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, ]))

        self.instance_features_upsample = build_upsample_layer(upsample_cfg_)
        
        #here...............
        self.MDC = MDC(channel_dim=256+256+2,out_channel=256)

        kernel = torch.FloatTensor([[0, 1, 0],
                                    [1, 100, 1],
                                    [0, 1, 0]]).unsqueeze(0).unsqueeze(0) #shape [1,1,3,3]
        self.filter_ = nn.Parameter(data=kernel, requires_grad=False)

    def get_edge_target(self,mask_target):
        mask_target = mask_target[:, None, :, :] #shape [n,1,mask_size,mask_size]
        edge_gt = F.conv2d(mask_target, self.filter_, padding=1)
        edge_gt_new = torch.zeros_like(edge_gt)
        edge_gt_id = ((edge_gt > 99) & (edge_gt < 104))
        edge_gt_new[edge_gt_id] = 1
        edge_gt_w_id = (edge_gt_new == 1)
        edge_gt_w = torch.ones_like(edge_gt)
        edge_gt_w[edge_gt_w_id] = 5
        return edge_gt_new, edge_gt_w


    def init_weights(self):
        pass
        """
        for m in self.modules():
            if m is None:
                continue
            elif isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        """
        """
        for m in [self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        """

    @auto_fp16()
    def forward(self, instance_feats,semantic_feat,rois): #x,x[0]--shape(b,256,w,h),rois
        for conv in self.convs:
            instance_feats = conv(instance_feats) #4 size=14
            
        instance_feat_28 = self.instance_features_upsample(instance_feats)
        mask_28 = self.mask28_out(instance_feat_28) #shape [n,1,28,28]

        #for conv in self.semantic_convs: #lowest feature extract 2*conv(3,3)
        #    semantic_feat = conv(semantic_feat) #size=28
        """
        semantic_out = self.semantic_out(semantic_feat) #shape [2,1,w,h] lowest feature  predict segmantic output
        
        semantic_roi_extrac = self.semantic_roi_extractor([semantic_feat,],rois)  #28x28 roi feature in low level  error !!!!!!!!!!!!!!!!
        
        
        mask_28 = self.mask28_out(instance_feat_28) #shape [n,1,28,28]
        edge_28 = self.edge28_out(instance_feat_28) #shape [n,1,28,28]
        
        concat_list = [instance_feat_28,semantic_roi_extrac,mask_28,edge_28] #channel 256,256,1,1
        
        cat_feature = torch.cat(concat_list,dim=1)
        
        fusion_feature = self.MDC(cat_feature)   #size=28 channel 256
       

        for conv in self.fusion_feature_convs: #fusion feature cross 2 convs
            fusion_feature = conv(fusion_feature)

        
        featurs_56 = self.relu(self.upsample(fusion_feature)) #shape [b,256,56,56]
        
        mask_56 = self.mask56_out(featurs_56) #shape [n,1,56,56]
        edge_56 = self.edge56_out(featurs_56)#self.edge_conv_56(featurs_56)) #shape [n,1,56,56]
        """
        
        return_list = []
        #return_list.append(semantic_out)
        #return_list.append(edge_28)
        return_list.append(mask_28)
        #return_list.append(edge_56)
        #return_list.append(mask_56)
       
        return return_list

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
    
        #get mask targets and edge target
        
        mask_28_dict = ConfigDict(mask_size=28)
        mask_56_dict = ConfigDict(mask_size=56)
        
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        
        mask_targets_28 = mask_target(pos_proposals, pos_assigned_gt_inds,#mask_targrt.shape = [n,28,28]
                                   gt_masks, mask_28_dict)
        mask_targets_56 = mask_target(pos_proposals, pos_assigned_gt_inds,#mask_targrt.shape = [n,28,28]
                                   gt_masks, mask_56_dict)

        edge_targets_28 = self.get_edge_target(mask_targets_28)
        edge_targets_56 = self.get_edge_target(mask_targets_56)


        # print(mask_targets)
        segmantic_list = []
        for gt_mask in gt_masks:
            segmantic, _ = torch.from_numpy(gt_mask.to_ndarray()).max(dim=0, keepdim=True)

            segmantic= segmantic.to(device=sampling_results[0].bboxes.device,
                dtype=torch.float32)
            segmantic_list.append(segmantic)
        segmantic_target = torch.cat(segmantic_list, dim=1)
        mask_targets = []
        #mask_targets.append(segmantic_target)
        #mask_targets.append(edge_targets_28)
        mask_targets.append(mask_targets_28)
        #mask_targets.append(edge_targets_56)
        #mask_targets.append(mask_targets_56) #与输出一一对应
        

        return mask_targets

    def mutil_loss(self,mask_pred,mask_targets,labels):
        """
                mask_pred --[28mask,28edge(target,weight),56mask,56edge(target,weight),segmantic]

                mask_targets -[28mask,28edge(target,weight),56mask,56edge(target,weight),segmantic]

        """
        loss = dict()
        #assert mask_pred[0].shape[-1]==mask_targets[0].shape[-1]
        
        #loss_semantic = F.binary_cross_entropy_with_logits(mask_pred[0].squeeze(0),mask_targets[0])

        #loss_edge28 = F.binary_cross_entropy_with_logits(mask_pred[1],mask_targets[1][0],weight=mask_targets[1][1])
        

        loss_mask28 = self.loss_mask(mask_pred[0],mask_targets[0],labels)

        #loss_edge56 = F.binary_cross_entropy_with_logits(mask_pred[3],mask_targets[3][0],weight=mask_targets[3][1])

        #loss_mask56 = F.binary_cross_entropy_with_logits(mask_pred[4].squeeze(),mask_targets[4])

        loss_mask = loss_mask28
      
        #loss_edge = loss_edge56
      
        #loss.update(loss_semantic=loss_semantic)
        
        loss.update(loss_mask=loss_mask)
      
        #loss.update(loss_edge=loss_edge)
    

        # if mask_pred.size(0) == 0:
        #     loss_mask = mask_pred.sum()
        # else:
        #     if self.class_agnostic:
        #         loss_mask = self.loss_mask(mask_pred, mask_targets,
        #                                    torch.zeros_like(labels))
        #     else:
        #         loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
       
        """
        assert len(maskbrance_pred)==len(maskbrance_targets)
        
        mask_pred_28 = [maskbrance_pred[0],maskbrance_pred[1]]
        mask_target_28 = [maskbrance_targets[0],maskbrance_targets[1]]
        
        mask_pred_56 = [maskbrance_pred[2],maskbrance_pred[3]]
        mask_target_56 = [maskbrance_targets[2],maskbrance_targets[3]]
        seg_labels = torch.zeros(len(maskbrance_pred[4])).long()
     
        #print(maskbrance_pred[4].shape)
        #print(maskbrance_targets[4].shape)
        #assert maskbrance_pred[4].shape==maskbrance_targets[4].shape
        loss_segmantic = self.loss_segmantic(maskbrance_pred[4], maskbrance_targets[4],seg_labels) #number tensor
        
        loss_mask_28,loss_edge_28 = self.mask_edge_loss(mask_pred_28,mask_target_28,labels)
        

        loss_mask_56,loss_edge_56 = self.mask_edge_loss(mask_pred_56,mask_target_56,labels)
        
        loss_mask = loss_mask_28 + loss_mask_56
        loss_edge = loss_edge_28 + loss_edge_56
        loss_segmantic = loss_segmantic*self.loss_weight[0]
        
        #print(loss_mask)
        #print(loss_segmantic)

        loss.update(loss_mask=loss_mask)
        loss.update(loss_edge=loss_edge)
        loss.update(loss_segmantic=loss_segmantic)
        """
        
        
        return loss

    def mask_edge_loss(self,pred,target,lable):

        mask_pred = pred[0]
        edge_pred = pred[1]

        mask_target = target[0]
        edge_target = target[1][0]
        edge_weigts = target[1][1]
        #print(mask_pred)
        #print(mask_target)
        loss_mask = self.loss_mask(mask_pred,mask_target,lable)
        loss_edge = self.loss_edge(edge_pred,edge_target,edge_weigts)

        return loss_mask,loss_edge

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
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

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes. #scale_factor :tensor([0.3200,0.3200,0.3200,0.3200])

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
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

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor #

        if torch.onnx.is_in_onnx_export():
            # TODO: Remove after F.grid_sample is supported.
            from torchvision.models.detection.roi_heads \
                import paste_masks_in_image
            masks = paste_masks_in_image(mask_pred, bboxes, ori_shape[:2])
            thr = rcnn_test_cfg.get('mask_thr_binary', 0)
            if thr > 0:
                masks = masks >= thr
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
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks) #torch.chunk鎷嗗垎tensor

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
                skip_empty=device.type == 'cpu') #skip_empty = false

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
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W .........(N,1,28,28)
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted. 1200
        img_w (int): Width of the image to be pasted. 1600
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
        x1_int, y1_int = img_w, img_h #1200,1600鐨刡itmap鐭╅樀
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    if torch.onnx.is_in_onnx_export():
        raise RuntimeError(
            'Exporting F.grid_sample from Pytorch to ONNX is not supported.')
    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
