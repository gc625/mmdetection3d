# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .votenet import VoteNet
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
import torch
@DETECTORS.register_module()
class DETR3D_MULTI(VoteNet):

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(DETR3D_MULTI, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

    
    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        # points_cat = torch.stack(points)
        points = points.unsqueeze(0)
        img_metas = [img_metas]
        x = self.extract_feat(points)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        exit()
        return bbox_results