from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS
from losses.detr3d_loss import build_criterion
@HEADS.register_module()
class DETR3DBboxHead(BaseModule):

    def __init__(self, 
    query_xyz,
    point_cloud_dims,
    box_features,
    init_cfg = None):
        super().__init__(init_cfg=init_cfg)

        self.query_xyz = query_xyz
        self.point_cloud_dims = point_cloud_dims
        self.box_features = box_features


    #TODO: 

    def loss(self):
        pass

    def get_bboxes(self):
        pass


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
                box_corners = self.box_processor.box_parametrization_to_corners(
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


    def forward(self):
        outputs = self.get_box_predictions(
            self.query_xyz,
            self.point_cloud_dims,
            self.box_features)

        

