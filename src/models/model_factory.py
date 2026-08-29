from models.Multiscale_ERT_Detr import MultiDurationRTDetrAttention as RtDetrMultiScaleAttention

def get_model(model_type, device, config):
    """
    Factory function to get the model based on the model type name.
    Returns an instance of the model class.
    """
    if model_type == 'rtdetrv2_multiscale':
        model = RtDetrMultiScaleAttention(
            checkpoint="PekingU/rtdetr_v2_r18vd",
            num_labels=1,
            durations_ms=getattr(config, 'durations', [3, 33, 165, 330]),
        )
    elif model_type == 'rtdetrv2':
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
    
        ID_TO_CLASS = {0: 'drone'}
        CLASS_TO_ID = {'drone': 0}
        
        config = RTDetrV2Config.from_pretrained(
            "PekingU/rtdetr_v2_r18vd",
            num_labels=len(ID_TO_CLASS),
            id2label=ID_TO_CLASS,
            label2id=CLASS_TO_ID,
            num_queries=config.queries,
            vfl_loss_coefficient=config.vfl_loss_coefficient,
        )
        
        model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_v2_r18vd",
            config=config,
            ignore_mismatched_sizes=True, 
        )         

        print(f'model.config.num_queries: {model.config.num_queries}')
    
    elif model_type == 'rtdetrv2_past_conditioned':
        model = RtDetrMultiScaleAttention(
            checkpoint="PekingU/rtdetr_v2_r18vd",
            num_labels=1,
            durations_ms=getattr(config, 'durations', [3, 33, 165, 330]),
            num_past_steps=getattr(config, 'num_past_annotations', 10),
            phase=getattr(config, 'phase', 1),
            num_standard_queries=getattr(config, 'num_standard_queries', 50),
            num_future_steps=getattr(config, 'num_future_steps', 0),
            forecast_head_type=getattr(config, 'forecast_head_type', 'transformer'),
            vel_avg_k=getattr(config, 'vel_avg_k', 3),
            use_cv_anchor=bool(getattr(config, 'use_cv_anchor', 1)),
            use_present_refine=bool(getattr(config, 'use_present_refine', 1)),
            use_shared_weights=bool(getattr(config, 'use_shared_weights', 0)),
            shared_cat=bool(getattr(config, 'shared_cat', 0)),
            block_diag_decoder_attn=bool(getattr(config, 'block_diag_decoder_attn', 0)),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    model.to(device)
    return model
    
