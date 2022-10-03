model = dict(
    type = '3DETR',
    backbone = dict(
        type = '3DETR_BACKBONE',
        # enc_type = 'maskedv2', # choices=["masked", "maskedv2", "vanilla"]
        # Vanilla encoder args


        preenc_dict = dict(
            use_color = False,
            enc_dim = 256, # same 
            preenc_npoints = 2048
        ),

        encoder_dict = dict(
            enc_type = 'maskedv2',
            enc_dim = 256, # same
            enc_nhead = 4,
            enc_ffn_dim = 128,
            enc_dropout = 0.1,
            enc_activateion = 'relu',
            enc_nlayers = 3,
            enc_pos_embed = None
        ),

        decoder_dict = dict(
            dec_dim = 256,
            dec_nhead = 4,
            dec_ffn_dim = 256,
            dec_dropout = 0.1,
            dec_nlayers = 8
        )

        encoder_dim = 256 #enc_dim
        decoder_dim = 256 #dec_dim
        mlp_dropout = 0.3 #mlp_dropout
        num_queries = 256 # nqueries



        # enc_nlayers = 3,
        # enc_dim = 256,
        # enc_ffn_dim = 128,
        # enc_dropout = 0.1,
        # enc_nhead = 4,
        # enc_pos_embed = None,
        # enc_activation = 'relu',
        # Decoder
        # dec_nlayers = 8,
        # dec_dim = 256,
        # dec_ffn_dim = 256,
        # dec_dropout = 0.1,
        # dec_nhead = 4,
        # other model params
        # preenc_npoints = 2048,
        pos_embed = 'fourier', # choices=["fourier", "sine"]
        nqueries = 256,
        # use_color = False, # DONT NEED?   


    ),
    bbox_head = dict(
        type = 'BoxProcessor', # 3detr boxprocessor
        
        # MLP Heads for bounding boxes
        mlp_droupout = 0.3,
        nsemcls = -1, # inferred from dataset

        # MATCHER
        matcher_giou_cost = 2,
        matcher_cls_cost = 1,
        matcher_center_cost = 0,
        matcher_objectness_cost = 0,

        # LOSS WEIGHTS 
        loss_giou_weight = 0,
        loss_sem_cls_weight = 1,
        loss_no_object_weight = 0.2,
        loss_angle_cls_weight = 0.1,
        loss_angle_reg_weight = 0.5,
        loss_center_weight = 5.0,
        loss_size_weight = 1.0 
    ),
    train_cfg = dict(
        # stuff
    ),
    test_cfg = dict(
        # stuff
    )
)