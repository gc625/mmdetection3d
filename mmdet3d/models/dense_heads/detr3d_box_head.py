from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS
from mmdet3d.models.losses.detr3d_loss import build_criterion
from mmdet3d.ops.detr3d_modules.helpers import GenericMLP
from mmdet3d.ops.detr3d_utils.pc_util import scale_points, shift_scale_points
from mmdet3d.ops.detr3d_utils.box_util import flip_axis_to_camera_np, flip_axis_to_camera_tensor, get_3d_box_batch_np, get_3d_box_batch_tensor,flip_lidar_axis_to_camera_tensor
import numpy as np
import torch.nn as nn
from functools import partial
import torch



class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, num_angle_bin,num_semcls):
        self.num_angle_bin = num_angle_bin
        self.num_semcls = num_semcls

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def dataset_box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_lidar_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes


    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


@HEADS.register_module()
class DETR3DBboxHead(BaseModule):

    def __init__(
        self, 
        mlp_dropout,
        num_semcls,
        matcher_giou_cost,
        matcher_cls_cost,
        matcher_center_cost,
        matcher_objectness_cost,
        loss_giou_weight,
        loss_sem_cls_weight,
        loss_no_object_weight,
        loss_angle_cls_weight,
        loss_angle_reg_weight,
        loss_center_weight,
        loss_size_weight,
        num_angle_bin,
        decoder_dim,
        train_cfg = None,
        test_cfg = None,
        init_cfg = None
        ):
        super().__init__(init_cfg=init_cfg)

        self.num_semcls = num_semcls
        self.num_angle_bin = num_angle_bin
        
        self.box_processor = BoxProcessor(num_angle_bin,num_semcls)
        self.build_mlp_heads(decoder_dim, mlp_dropout)
        self.criterion = build_criterion(
            matcher_cls_cost,
            matcher_giou_cost,
            matcher_center_cost,
            matcher_objectness_cost,
            loss_giou_weight,
            loss_sem_cls_weight,
            loss_no_object_weight,
            loss_angle_cls_weight,
            loss_angle_reg_weight,
            loss_center_weight,
            loss_size_weight,
            num_semcls,
            num_angle_bin)

    #TODO: 
    def build_mlp_heads(self, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=self.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=self.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=self.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)


    
    def loss(self,outputs,ret_dict):
        loss, loss_dict = self.criterion(outputs,ret_dict)
        return loss, loss_dict



    def get_box_predictions(
        self, 
        query_xyz, 
        point_cloud_dims, 
        box_features):
            """
            Parameters:
                query_xyz: batch x nqueries x 3 tensor of query XYZ coords
                point_cloud_dims: List of [min, max] dims of point cloud
                                min: batch x 3 tensor of min XYZ coords
                                max: batch x 3 tensor of max XYZ coords
                box_features: num_layers x num_queries x batch x channel
            """
            # box_features change to (num_layers x batch) x channel x num_queries
            box_features = box_features.permute(0, 2, 3, 1)
            num_layers, batch, channel, num_queries = (
                box_features.shape[0],
                box_features.shape[1],
                box_features.shape[2],
                box_features.shape[3],
            )
            box_features = box_features.reshape(num_layers * batch, channel, num_queries)

            # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
            cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
            center_offset = (
                self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
            )
            size_normalized = (
                self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
            )
            angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
            angle_residual_normalized = self.mlp_heads["angle_residual_head"](
                box_features
            ).transpose(1, 2)

            # reshape outputs to num_layers x batch x nqueries x noutput
            cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
            center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
            size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
            angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
            angle_residual_normalized = angle_residual_normalized.reshape(
                num_layers, batch, num_queries, -1
            )
            angle_residual = angle_residual_normalized * (
                np.pi / angle_residual_normalized.shape[-1]
            )

            outputs = []
            for l in range(num_layers):
                # box processor converts outputs so we can get a 3D bounding box
                (
                    center_normalized,
                    center_unnormalized,
                ) = self.box_processor.compute_predicted_center(
                    center_offset[l], query_xyz, point_cloud_dims
                )
                angle_continuous = self.box_processor.compute_predicted_angle(
                    angle_logits[l], angle_residual[l]
                )
                size_unnormalized = self.box_processor.compute_predicted_size(
                    size_normalized[l], point_cloud_dims
                )
                box_corners = self.box_processor.dataset_box_parametrization_to_corners(
                    center_unnormalized, size_unnormalized, angle_continuous
                )

                # below are not used in computing loss (only for matching/mAP eval)
                # we compute them with no_grad() so that distributed training does not complain about unused variables
                with torch.no_grad():
                    (
                        semcls_prob,
                        objectness_prob,
                    ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

                box_prediction = {
                    "sem_cls_logits": cls_logits[l],
                    "center_normalized": center_normalized.contiguous(),
                    "center_unnormalized": center_unnormalized,
                    "size_normalized": size_normalized[l],
                    "size_unnormalized": size_unnormalized,
                    "angle_logits": angle_logits[l],
                    "angle_residual": angle_residual[l],
                    "angle_residual_normalized": angle_residual_normalized[l],
                    "angle_continuous": angle_continuous,
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners,
                }
                outputs.append(box_prediction)

            # intermediate decoder layer outputs are only used during training
            aux_outputs = outputs[:-1]
            outputs = outputs[-1]

            return {
                "outputs": outputs,  # output from last layer of decoder
                "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
            }


    def forward(self,query_xyz,point_cloud_dims,box_features):
        outputs = self.get_box_predictions(
            query_xyz,
            point_cloud_dims,
            box_features)

        return outputs

        
    
