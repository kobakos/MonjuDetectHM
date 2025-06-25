import os
import sys
import time
import random
from pathlib import Path
from collections import OrderedDict

import cv2
import yaml
import timm
import torch
import copick
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from monjudetecthm.data_processing import build_dataloaders
from monjudetecthm.data_processing.dataset import generate_sliding_window_index
from monjudetecthm.utils.loop import to_device
from monjudetecthm.models import build_model_from_config
from monjudetecthm.models.postprocessing import PostProcessor
from monjudetecthm.evaluation import score, pred_dicts_to_df, generate_gt_sub_df

from monjudetecthm.utils import sigmoid
from monjudetecthm.utils import load_configs

# set random seed for everything
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

transformations = {
    'e': lambda x: x,
    'b': lambda x: x.flip(dims=(-2,)),
    'ba2': lambda x: x.flip(dims=(-1,)),
    'a2': lambda x: x.flip(dims=(-1, -2)),
    'a': lambda x: x.rot90(dims=(-1, -2), k=1),
    'a3': lambda x: x.rot90(dims=(-1, -2), k=3),
    'ba': lambda x: x.flip(dims=(-2,)).rot90(dims=(-1, -2), k=1),
    'ba3': lambda x: x.flip(dims=(-2,)).rot90(dims=(-1, -2), k=3),
    'c': lambda x: x.flip(dims=(-3,)),
    'cb': lambda x: x.flip(dims=(-3, -2)),
    'cba2': lambda x: x.flip(dims=(-3, -1)),
    'ca2': lambda x: x.flip(dims=(-3, -2, -1)),
    'ca': lambda x: x.flip(dims=(-3,)).rot90(dims=(-2, -1), k=1),
    'ca3': lambda x: x.flip(dims=(-3,)).rot90(dims=(-2, -1), k=3),
    'cba': lambda x: x.flip(dims=(-3, -2)).rot90(dims=(-1, -2), k=1),
    'cba3': lambda x: x.flip(dims=(-3, -2)).rot90(dims=(-1, -2), k=3),
}

reverse_transformations = {
    'e': 'e',
    'b': 'b',
    'ba2': 'ba2',
    'a2': 'a2',
    'a': 'a3',
    'a3': 'a',
    'ba': 'ba',
    'ba3': 'ba3',
    'c': 'c',
    'cb': 'cb',
    'cba2': 'cba2',
    'ca2': 'ca2',
    'ca': 'ca3',
    'ca3': 'ca',
    'cba': 'cba',
    'cba3': 'cba3',
}

def infer_transformerd(model, images, transformation):
    out = model(transformations[transformation](images))
    if isinstance(out, tuple):
        out = (
            transformations[reverse_transformations[transformation]](out[0]),
            transformations[reverse_transformations[transformation]](out[1])
        )
    else:
        out = transformations[reverse_transformations[transformation]](out)
    
    return out

def infer(models, images, TTA):
    out = 0
    with torch.inference_mode():
        for model in models:
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                for t in TTA:
                    out += infer_transformerd(model, images, t).float()
        out /= len(models)
        out /= len(TTA)
    return out

def TTA_avg(models, pbar, val_df, val_index, TTA, cfg, copick_root):
    start = time.time()
    time_postrocess = 0
    time_inference = 0
    bs = cfg['train_loop']['val_batch_size']
    post_processor = PostProcessor(
        copick_root=copick_root,
        window_size=cfg['dataset']['image_size'],
        window_stride=cfg['dataset']['image_stride'],
        voxel_spacing=cfg['dataset']['voxel_spacing'],
        **cfg.get("postprocessing", {})
    )
    for i, (images, targets) in enumerate(pbar):
        images, targets = to_device(images, targets, cfg['system']['device'])

        out = infer(models, images, TTA)

        crop_origins = val_df.loc[val_index[i * bs: (i + 1) * bs], "crop_origin_d0":"crop_origin_d2"].values
        experiment_ids = val_df.loc[val_index[i * bs: (i + 1) * bs], 'experiment_id'].values
        post_processor.accumulate(out, crop_origins, experiment_ids)
    assert post_processor.accumulated_data == {}
    end = time.time()
    return post_processor, end-start, time_postrocess, time_inference

def load_models(model_config_paths, weight_paths):
    """
    Load models from the given configuration and weight paths.
    Args:
        model_config_paths (list of list of str): List of lists containing paths to model configuration files.
        weight_paths (list of list of str): List of lists containing paths to model weight files.
    The most outer list corresponds to the model ensemble, the inner lists correspond to the model weight averaging.
    """
    models = []
    for ps, mcfgs in zip(weight_paths, model_config_paths):
        models_i = []
        for p, mcfg in zip(ps, mcfgs):
            with open(mcfg, encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)["model"]
            if 'ema' in model_cfg:
                del model_cfg['ema']
            model_cfg['pretrained'] = False
            if 'pretrain_weight_path' in model_cfg:
                del model_cfg['pretrain_weight_path']
            model = build_model_from_config(model_cfg)
            state_dict = torch.load(p, map_location=cfg['system']['device'])
            if '_orig_mod.segmentation_head.weight' in state_dict:
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            del state_dict
            model.eval()
            models_i.append(model)
        model = models_i[0]
        for k in model.state_dict():
            model.state_dict()[k] = sum([m.state_dict()[k] for m in models_i])/len(models_i)
        models.append(model)
    return models

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--copick_config', type=str, help='Path to the copick config file', default='copick_config.json')
    parser.add_argument('--config', type=str, help='Path to the inference config file', default='configs/infer_config.yml')
    parser.add_argument('--model_path', type=str, help='Path to the directory containing configs and weights', default='results/209_resnet50d.ra2_in1k_20250116')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enable anomaly detection')
    parser.add_argument('--tqdm', action='store_true', help='Use tqdm')
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, encoding='utf-8')as f:
        cfg = yaml.safe_load(f)

    copick_root = copick.from_file(args.copick_config)

    val_df = generate_sliding_window_index(
        copick_root=copick_root,
        voxel_spacing=cfg['dataset']['voxel_spacing'],
        image_size=cfg['dataset']['image_size'],
        image_stride=cfg['dataset']['image_stride'],
        include_edge_windows=True,
    )

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    if cfg['system']['infer_amp_dtype'] == 'fp32':
        AMP_DTYPE = torch.float32
    elif cfg['system']['infer_amp_dtype'] == 'bf16':
        AMP_DTYPE = torch.bfloat16
    elif cfg['system']['infer_amp_dtype'] == 'fp16':
        AMP_DTYPE = torch.float16
    else:
        raise ValueError(f'Unknown dtype: {cfg["system"]["infer_amp_dtype"]}')
    
    seed_everything(cfg['system']['seed'])
    
    Transformations=[
        'e', # no transformation
        #'b', #flip1
        #'ba2', # flip2
        #'a2', # flip12,
        #'a', # rot12
        #'a3', # rot12*3
        #'ba', # flip1, rot12
        #'ba3', # flip1, rot12*3
        #'c', # flip0
        #'cb', # flip0, flip1
        #'cba2', # flip0, flip2
        #'ca2', # flip0, flip12,
        #'ca', # flip0, rot12
        #'ca3', # flip0, rot12*3
        #'cba', # flip0, flip1, rot12
        #'cba3', # flip0, flip1, rot12*3
    ]

    torch.cuda.empty_cache()
    
    start = time.time()
    time_postrocess = 0
    time_inference = 0

    # Search for all .pth and .pt weight files
    model_path = Path(args.model_path)
    
    # Search for all config files (.yml, .yaml)
    weight_paths = [[model_path / 'weights' / f'fold_{i}' / 'best.pth' for i in range(4)]]
    model_config_paths = [[model_path / "config.yml"]*4]

    weight_paths = [[weight_paths[0][0]]]
    model_config_paths = [[model_config_paths[0][0]]]
    
    models = load_models(
        model_config_paths=model_config_paths,
        weight_paths=weight_paths,
    )

    val_index = val_df.index
    train_dl, val_dl = build_dataloaders(
        copick_root=copick_root,
        df=val_df,
        train_index=val_index,# placeholder for train_index, not used in inference
        val_index=val_index,
        cfg=cfg,
    )
    val_sub_df = generate_gt_sub_df(
        copick_root=copick_root
    )
    
    val_loss_mean = 0
    if args.tqdm:
        pbar = tqdm(val_dl, desc=f'Validation', total=len(val_dl), leave=False)
    else:
        pbar = val_dl
    
    post_processor, total_time, total_time_inference, total_time_postprocess = TTA_avg(
        models=models,
        pbar=pbar,
        val_df=val_df,
        val_index=val_index,
        TTA=Transformations,
        cfg=cfg,
        copick_root=copick_root,
    )
    preds = post_processor.predictions

    pred_sub_df = pred_dicts_to_df(preds)
    val_metric_mean = score(val_sub_df, pred_sub_df, distance_multiplier=0.5)
    print(f'avg Validation Loss: {val_loss_mean: .5e}, avg Validation Metric: {val_metric_mean: .5f}')
    print()
    del models
    torch.cuda.empty_cache()
    