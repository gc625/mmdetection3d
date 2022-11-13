# Copyright (c) Facebook, Inc. and its affiliates.
from json import encoder
import math
from functools import partial
from ..builder import BACKBONES
import numpy as np
import torch
import torch.nn as nn
from mmdet3d.ops.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from mmdet3d.ops.pointnet2.pointnet2_utils import furthest_point_sample
from mmdet3d.ops.detr3d_utils.pc_util import scale_points, shift_scale_points

from mmdet3d.ops.detr3d_modules.helpers import GenericMLP
from mmdet3d.ops.detr3d_modules.position_embedding import PositionEmbeddingCoordsSine
from mmdet3d.ops.detr3d_modules.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer,MultiMaskedTransformerEncoder)

from mmdet3d.ops import build_sa_module


def build_preencoder(args):
    enc_dim = args['enc_dim']
    preenc_npoints = args['preenc_npoints']
    radius = args['radius']
    nsample = args['nsample']

    mlp_dims = [1, 64, 128, enc_dim]


    preencoder = PointnetSAModuleVotes(
        radius=radius,
        nsample=nsample,
        npoint=preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    
    enc_type = args['enc_type']
    enc_dim = args['enc_dim']
    enc_nhead = args['enc_nhead']
    enc_ffn_dim = args['enc_ffn_dim'] 
    enc_dropout = args['enc_dropout'] 
    enc_activation = args['enc_activation']
    enc_nlayers = args['enc_nlayers'] 
    preenc_npoints = args['preenc_npoints']
    # interim_indices = args['interim_indices']
    
    interim_dict = args['interim_dict']
    num_points = interim_dict['num_points']
    radii = interim_dict['radii']
    num_samples = interim_dict['num_samples']
    sa_channels = interim_dict['sa_channels']
    fps_mods = interim_dict['fps_mods']
    fps_sample_range_lists = interim_dict['fps_sample_range_lists']    
    dilated_group = interim_dict['dilated_group']
    norm_cfg = interim_dict['norm_cfg']
    sa_cfg = interim_dict['sa_cfg']


    sa_layers = []
    for sa_index in range(len(num_points)):
        cur_sa_mlps = list(sa_channels[sa_index])
        cur_fps_mod = list([fps_mods[sa_index]])
        cur_fps_sample_range_list = list([fps_sample_range_lists[sa_index]])
        
        sa_layer = build_sa_module(
                        num_point=num_points[sa_index],
                        radii=radii[sa_index],
                        sample_nums=num_samples[sa_index],
                        mlp_channels=cur_sa_mlps,
                        fps_mod=cur_fps_mod,
                        fps_sample_range_list=cur_fps_sample_range_list,
                        dilated_group=dilated_group[sa_index],
                        norm_cfg=norm_cfg,
                        cfg=sa_cfg,
                        bias=True)
        sa_layers += [sa_layer]


    encoder_layers = []

    for i in range




    


    
    
    
    encoder_layer = TransformerEncoderLayer(
        d_model = enc_dim,
        nhead = enc_nhead,
        dim_feedforward= enc_ffn_dim,
        dropout = enc_dropout,
        activation = enc_activation
    )    


    interim_downsampling = PointnetSAModuleVotes(
        radius=0.4,
        nsample=32,
        npoint= preenc_npoints // 2,
        mlp=[ enc_dim, 256, 256, enc_dim],
        normalize_xyz=True,
    )


    



    masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
    # encoder = MaskedTransformerEncoder(
    #     encoder_layer=encoder_layer,
    #     num_layers=enc_nlayers,
    #     interim_downsampling=interim_downsampling,
    #     masking_radius=masking_radius,
    # )

    
    encoder = MultiMaskedTransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=enc_nlayers,
    interim_downsampling=interim_downsampling,
    masking_radius=masking_radius,
    interim_indices=interim_indices
    )





@BACKBONES.register_module()
class DETR3D_multiscale_backbone(nn.Module):

    def __init__(
        self,
        preenc_dict,
        encoder_dict,
        decoder_dim=256,
        num_queries=256):
        positional_embedding="fourier",
        super().__init__()

        self.pre_encoder = build_preencoder(preenc_dict)
        self.encoder = build_encoder(encoder_dict)
        self.num_queries = num_queries



        # self.pos_embedding = PositionEmbeddingCoordsSine(
        #     d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        # )        
    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds.to(torch.int64), 1, enc_inds.to(torch.int64))
        return enc_xyz, enc_features, enc_inds


    def forward(self, inputs, img=None,encoder_only=False):
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        

