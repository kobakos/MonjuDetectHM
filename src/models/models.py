import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import timm_3d
from timm.utils.model_ema import ModelEmaV2
from src.models.prepare_3d_model import TimmSegModel


def build_model_from_config(cfg):
    model = TimmSegModel(
        backbone = cfg['name'],
        decoder_channels = cfg['decoder_channels'],
        drop_path_rate = cfg['drop_path_rate'],
        drop_rate = cfg['drop_rate'],
        final_activation = cfg['final_activation'],
        n_classes = cfg['n_classes'],
        pretrained = cfg['pretrained'],
        use_timm_3d = cfg['use_timm_3d'],
        decoder_attention_type=cfg.get('decoder_attention_type', None),
        dann = cfg.get('dann', False),
        dann_scheduler_cfg = cfg.get('dann_schedule', {}),
        architecture = cfg.get('architecture', 'unet'),
        input_upsample=cfg.get('input_upsample', 1.0),
        n_blocks=cfg.get('n_blocks', 4),
        w_offset_head=cfg.get('w_offset_head', False),
        output_stride=cfg.get('output_stride', None),
        aspp_dilation=cfg.get('aspp_dilation', [6, 12, 18]),
    )
    model.to(cfg['device'])
    if 'pretrain_weight_path' in cfg:
        try: 
            model.load_state_dict(torch.load(cfg['pretrain_weight_path'], map_location=cfg['device']), strict=True)
        except RuntimeError:
            print('Loading pretrain weight with strict=False')
            model.load_state_dict(torch.load(cfg['pretrain_weight_path'], map_location=cfg['device']), strict=False)
            
    if 'ema' in cfg:
        model = ModelEmaV2(
            model,
            decay=cfg['ema']['decay'],
            device=cfg['device'],
        )

    if 'compile' in cfg and cfg['compile']:
        model = torch.compile(model)

    return model

if __name__ == "__main__":
    import torch
    import yaml

    with open('../../configs/config.yml', 'r') as f:
        cfg = yaml.safe_load(f)
                        
    net = build_model_from_config(cfg['model'])
    input = torch.zeros(2, 1, 128, 128, 128)
    print('input shape: ', input.shape)
    with torch.no_grad():
        output = net(input)
        
    print('output shape: ', output.shape)