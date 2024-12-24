from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from ..builder import build_shared_head


class PABaseRoIHead(BaseModule, metaclass=ABCMeta):
    """Base class for RoIHeads."""

    def __init__(self,
                 bbox_roi_extractor_lev1=None,
                 bbox_roi_extractor_lev2=None,
                 bbox_roi_extractor_lev3=None,
                 bbox_roi_extractor_lev4=None,
                 bbox_head=None,
                 mask_roi_extractor_lev1=None,
                 mask_roi_extractor_lev2=None,
                 mask_roi_extractor_lev3=None,
                 mask_roi_extractor_lev4=None,
                 #boundary_roi_extractor=None, #############
                 mask_head=None,
                 #as_head=None, ###here
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PABaseRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor_lev1,bbox_roi_extractor_lev2,bbox_roi_extractor_lev3,bbox_roi_extractor_lev4, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor_lev1,mask_roi_extractor_lev2,mask_roi_extractor_lev3,mask_roi_extractor_lev4, mask_head,)#boundary_roi_extractor)######################boundary_roi_extractor,
        """
        if as_head is not None:######
            self.init_as_head(as_head)
        """
        self.init_assigner_sampler()
        
        #if self.with_as:
        #    self.init_as_head()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None
    """
    @property
    def with_as(self):######
        
        return hasattr(self, 'as_head') and self.as_head is not None
    """
    

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass
  

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass

    @abstractmethod
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    @abstractmethod
    def forward_train(self,
                      x,
                      img_meta,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """Forward function during training."""

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Asynchronized test function."""
        raise NotImplementedError

    def simple_test(self,
                    x,
                    proposal_list,
                    img_meta,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
