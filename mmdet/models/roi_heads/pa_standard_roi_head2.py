import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core.bbox.iou_calculators import bbox_overlaps

@HEADS.register_module()
class PANetRoIHead(PABaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor_lev1,bbox_roi_extractor_lev2,bbox_roi_extractor_lev3,bbox_roi_extractor_lev4, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor1 = build_roi_extractor(bbox_roi_extractor_lev1)
        self.bbox_roi_extractor2 = build_roi_extractor(bbox_roi_extractor_lev2)
        self.bbox_roi_extractor3 = build_roi_extractor(bbox_roi_extractor_lev3)
        self.bbox_roi_extractor4 = build_roi_extractor(bbox_roi_extractor_lev4)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor_lev1,mask_roi_extractor_lev2,mask_roi_extractor_lev3,mask_roi_extractor_lev4, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor1 = build_roi_extractor(mask_roi_extractor_lev1)
            self.mask_roi_extractor2 = build_roi_extractor(mask_roi_extractor_lev2)
            self.mask_roi_extractor3 = build_roi_extractor(mask_roi_extractor_lev3)
            self.mask_roi_extractor4 = build_roi_extractor(mask_roi_extractor_lev4)
            
            self.share_roi_extractor = False
        else:  #there may be error
            self.share_roi_extractor = True  
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)
        
    def init_as_head(self,as_head):####here
        if as_head is not None:
            self.as_head=build_head(as_head)
        else:
            pass
    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # as_paent  #############here
        if self.with_as:
            count_num = self.as_head(x)
            count_num_target = self.as_head.get_target(gt_bboxes)
            loss_as = self.as_head.loss(count_num,count_num_target)
            losses.update(loss_count_num=loss_as)
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats1 = self.bbox_roi_extractor1(
            x[:self.bbox_roi_extractor1.num_inputs], rois)
        bbox_feats2 = self.bbox_roi_extractor2(
            x[:self.bbox_roi_extractor2.num_inputs], rois)
        bbox_feats3 = self.bbox_roi_extractor3(
            x[:self.bbox_roi_extractor3.num_inputs], rois)
        bbox_feats4 = self.bbox_roi_extractor4(
            x[:self.bbox_roi_extractor4.num_inputs], rois)
        """
        print(bbox_feats1.shape)
        print(bbox_feats2.shape)
        print(bbox_feats3.shape)
        print(bbox_feats4.shape)
        """
        bbox_feats = [bbox_feats1,bbox_feats2,bbox_feats3,bbox_feats4] #shape of every bbox_feats is 512×256×7×7
        
        # next step is feats fusion by max operation
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        #cls_score, bbox_pred ,iou_pred= self.bbox_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        ###########
        
        
        reg_delta = bbox_results['bbox_pred'] #batch=1 (512,4), batch=2(1024,4)
        target_delta = bbox_targets[2]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        '''
        if len(pos_bboxes_list)==1:
            #print('the len is 1')
            proposal_ = torch.cat((pos_bboxes_list[0],neg_bboxes_list[0]),dim=0)
        else:
            #print('the len is 2')
            proposal_ = torch.cat((pos_bboxes_list[0],neg_bboxes_list[0],pos_bboxes_list[1],neg_bboxes_list[1]),dim=0)
        GT_box =self.bbox_head.bbox_coder.decode(proposal_,target_delta)
        pred_box = self.bbox_head.bbox_coder.decode(proposal_,reg_delta)
        GT_box = torch.unsqueeze(GT_box,dim=0)
        pred_box = torch.unsqueeze(pred_box,dim=0)
        IOU_target = bbox_overlaps(GT_box,pred_box,mode='iou',is_aligned=True).transpose(1,0)
        '''

        ###########
        '''
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], bbox_results['iou_pred'],rois,
                                        *bbox_targets,IOU_target)
        '''
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],rois,
                                        *bbox_targets)
                                        

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats1 = self.mask_roi_extractor1(
                x[:self.mask_roi_extractor1.num_inputs], rois)
            mask_feats2 = self.mask_roi_extractor2(
                x[:self.mask_roi_extractor2.num_inputs], rois)
            mask_feats3 = self.mask_roi_extractor3(
                x[:self.mask_roi_extractor3.num_inputs], rois)
            mask_feats4 = self.mask_roi_extractor4(
                x[:self.mask_roi_extractor4.num_inputs], rois)
            #mask feats max fusion 
            mask_feats = [mask_feats1,mask_feats2,mask_feats3,mask_feats4,]
            
                
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.
        
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
       
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
