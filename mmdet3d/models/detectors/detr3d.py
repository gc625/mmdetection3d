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

        





    def forward_train(self, points,point_cloud_dims_min,point_cloud_dims_max, ret_dict,img_metas, **kwargs):
        input = {
            'point_clouds':torch.stack(points),
            'point_cloud_dims_min':point_cloud_dims_min,
            'point_cloud_dims_max':point_cloud_dims_max
        }
        query_xyz,point_cloud_dims,box_features = self.backbone(input)
        outputs = self.bbox_head(query_xyz,point_cloud_dims,box_features)
        loss = self.bbox_head.loss(outputs,ret_dict)        
        return 


    def simple_test(self, img, img_metas, **kwargs):
        return 

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)