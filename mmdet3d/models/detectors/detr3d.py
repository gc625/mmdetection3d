from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
import torch

@DETECTORS.register_module()
class DETR3D(SingleStage3DDetector):
    """
    3DETR 
    """

    def __init__(
        self, 
        backbone, 
        neck=None, 
        bbox_head=None, 
        train_cfg=None, 
        test_cfg=None, 
        init_cfg=None, 
        pretrained=None):
        
        super().__init__(
            backbone, 
            neck, 
            bbox_head, 
            train_cfg, 
            test_cfg, 
            init_cfg, 
            pretrained)

        





    def forward_train(self, points, img_metas, **kwargs):
        points_cat = torch.stack(points)
        feats = self.backbone(points_cat)

        
        return 


    def simple_test(self, img, img_metas, **kwargs):
        return 

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)