from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
from mmdet3d.ops.detr3d_utils.box_util import get_3d_box_batch_tensor
from mmdet3d.ops.detr3d_utils.ap_calculator import parse_predictions_mmdet
import torch
import numpy
import pickle 

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

        

    def bboxes_postprocess(self,outputs,point_cloud,img_metas):
        outputs = outputs["outputs"]
        predicted_box_corners=outputs["box_corners"],
        sem_cls_probs=outputs["sem_cls_prob"],
        objectness_probs=outputs["objectness_prob"],
        angle_continuous = outputs['angle_continuous']
        center_unnormalized = outputs['center_unnormalized']
        size_unnormalized = outputs['size_unnormalized']

        preds = parse_predictions_mmdet(
            predicted_box_corners, 
            sem_cls_probs, 
            objectness_probs,
            angle_continuous,
            center_unnormalized,
            size_unnormalized, 
            point_cloud,
            img_metas
)
        return [preds]


    def forward_train(self, points,point_cloud_dims_min,point_cloud_dims_max, ret_dict,img_metas, **kwargs):
        # torch.autograd.set_detect_anomaly(True)
        input = {
            'point_clouds':torch.stack(points),
            'point_cloud_dims_min':point_cloud_dims_min,
            'point_cloud_dims_max':point_cloud_dims_max
        }

        
        

        query_xyz,point_cloud_dims,box_features = self.backbone(input)
        outputs = self.bbox_head(query_xyz,point_cloud_dims,box_features)
        loss = self.bbox_head.loss(outputs,ret_dict)        
        return loss[1]


    def simple_test(self,points,img,img_metas,point_cloud_dims_min,point_cloud_dims_max,**kwargs):
        points =  torch.stack(points) if len(points.shape) > 2 else points.unsqueeze(axis=0)
        input = {
            'point_clouds': points,
            'point_cloud_dims_min':point_cloud_dims_min,
            'point_cloud_dims_max':point_cloud_dims_max
        }



        query_xyz,point_cloud_dims,box_features = self.backbone(input,img=img)
        outputs = self.bbox_head(query_xyz,point_cloud_dims,box_features)
        bbox_results = self.bboxes_postprocess(outputs,points,img)





        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)