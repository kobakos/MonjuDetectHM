import os
import sys
import random
from pathlib import Path
from collections import OrderedDict

import cv2
import yaml
import timm
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import copick
from copick.impl.filesystem import CopickRootFSSpec

from monjudetecthm.utils.utils import load_configs
from monjudetecthm.component_factory import create_optimizer, create_scheduler, create_criterion, create_metric
from monjudetecthm.data_processing.dataset import generate_sliding_window_index, build_dataloaders, split_by_run_name
from monjudetecthm.utils.loop import to_device
from monjudetecthm.utils import sigmoid
from monjudetecthm.models import build_model_from_config
from monjudetecthm.models.postprocessing import PostProcessor
from monjudetecthm.evaluation import score, pred_dicts_to_df, generate_gt_sub_df
from monjudetecthm.losses import from_cfg

def train_single_fold(fold, train_dl, val_dl, val_df, val_index, val_sub_df, cfg, copick_root, model_save_dir):
    print(f'============================================start of training=============================================')
    # create model and whatnot
    model = build_model_from_config(cfg['model'])
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    optimizer = create_optimizer(cfg['optimizer'], model.parameters())
    scheduler = create_scheduler(cfg['scheduler'], optimizer)
    criterion = from_cfg(cfg['criterion'])
    val_criterion = criterion
    post_processor = PostProcessor(
        copick_root=copick_root,
        window_size=cfg['dataset']['image_size'], 
        window_stride=cfg['dataset']['image_stride'],
        voxel_spacing=cfg['dataset']['voxel_spacing'],
        selected_classes=cfg['dataset']['classes'],
        **cfg["postprocessing"]
    )
    
    # main training loop
    best_val_loss = 1e6
    best_val_metric = 0
    for epoch in range(cfg['train_loop']['n_epochs']):
        # training loop
        model.train()
        all_losses_mean = 0
        #with tqdm(train_dl, desc=f'Epoch: {epoch}, Training', total=len(train_dl), leave=False) as pbar:
        if args.tqdm:
            pbar = tqdm(train_dl, desc=f'Epoch: {epoch}, Training', total=len(train_dl), leave=False)
        else:
            pbar = train_dl
        for i, (images, targets) in enumerate(pbar):
            images, targets = to_device(images, targets, cfg['system']['device'])
            with torch.autocast(device_type = cfg['system']['device'], dtype=AMP_DTYPE):
                out = model(images)
                loss = criterion(out, targets)

            # update various loss metrics
            all_losses_mean = (all_losses_mean * i + loss.item()) / (i + 1)

            loss.backward()
            if cfg['train_loop']['max_grad_norm'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_loop']['max_grad_norm'])
            
            if (i + 1) % cfg['train_loop']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            if cfg['scheduler']['name'] == 'cosine_timm':
                scheduler.step(epoch + i / len(train_dl))
            else:
                scheduler.step()

            if args.tqdm:          
                pbar.set_postfix(OrderedDict(
                    loss = loss.item(),
                    loss_mean = all_losses_mean
                ))
            if args.sanity_check:
                break
        print(f'Epoch: {epoch}, Loss_mean: {all_losses_mean: .5e}')
        if args.tqdm:
            pbar.close()
        model.eval()
        
        val_loss_mean = 0
        val_metric_mean = 0
        post_processor.reset()
        if args.tqdm:
            pbar = tqdm(val_dl, desc=f'Epoch: {epoch}, Validation', total=len(val_dl), leave=False)
        else:
            pbar = val_dl
        for i, (images, targets) in enumerate(pbar):
            images, targets = to_device(images, targets, cfg['system']['device'])
    
            with torch.no_grad():
                with torch.autocast(device_type=cfg['system']['device'], dtype=INFER_AMP_DTYPE):
                    out = model(images)
                    val_loss = val_criterion(out, targets)
            bs = cfg['train_loop']['val_batch_size']
            crop_origins = val_df.loc[val_index[i * bs: (i + 1) * bs], 'crop_origin_d0':'crop_origin_d2'].values
            experiment_ids = val_df.loc[val_index[i * bs: (i + 1) * bs], 'experiment_id'].values

            del images, targets
            post_processor.accumulate(out, crop_origins, experiment_ids)

            val_loss_mean = (val_loss_mean * i + val_loss.item()) / (i + 1)
        assert post_processor.accumulated_data == {}
        pred_sub_df = pred_dicts_to_df(post_processor.predictions)
        pred_sub_df.to_csv('predictions.csv', index=False)
        val_sub_df.to_csv('gt_predictions.csv', index=False)
        val_metric_mean = score(val_sub_df, pred_sub_df, None)
        if args.tqdm:
            pbar.close()
        best_val_loss = min(best_val_loss, val_loss_mean)
        best_val_metric = max(best_val_metric, val_metric_mean)
            
        print(f'Epoch: {epoch}, avg Validation Loss: {val_loss_mean: .5e}, avg Validation Metric: {val_metric_mean: .5f}')
        # Save model if it's the best so far
        save_by = cfg.get('save_by', 'val_metric')
        save_model = False
        if save_by == 'val_metric' and val_metric_mean >= best_val_metric:
            save_model = True
        elif save_by == 'val_loss' and val_loss_mean <= best_val_loss:
            save_model = True
        
        if save_model:
            fold_dir = os.path.join(model_save_dir, 'cryoet_detection_model', f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            model_save_path = os.path.join(fold_dir, 'best.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved best model to {model_save_path}')
        print()
        if args.sanity_check:
            break


if __name__ == '__main__':
    # set random seed for everything
    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    import argparse
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--config', type=str, help='Path to the config file (in json)', default='configs/config.yml')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enable anomaly detection')
    parser.add_argument('--tqdm', action='store_true', help='Use tqdm')
    parser.add_argument('--sanity_check', action='store_true', help='Run sanity check with short steps/epochs')
    parser.add_argument('--model_save_dir', type=str, default='results', help='Directory to save models')
    parser.add_argument('--copick_config_path', type=str, default='copick_config.json', help='Path to the CoPick config file')
    args = parser.parse_args()
    
    config_file = args.config
    with open(config_file, encoding='utf-8')as f:
        cfg = yaml.safe_load(f)

    if cfg["system"]['amp_dtype'] == 'fp16':
        AMP_DTYPE = torch.float16
    elif cfg["system"]['amp_dtype'] == 'bf16':
        AMP_DTYPE = torch.bfloat16
    elif cfg["system"]['amp_dtype'] == 'fp32':
        AMP_DTYPE = torch.float32
    elif cfg["system"]['amp_dtype'] == 'tf32':
        AMP_DTYPE = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg["system"]['infer_amp_dtype'] == 'fp16':
        INFER_AMP_DTYPE = torch.float16
    elif cfg["system"]['infer_amp_dtype'] == 'bf16':
        INFER_AMP_DTYPE = torch.bfloat16
    elif cfg["system"]['infer_amp_dtype'] == 'fp32':
        INFER_AMP_DTYPE = torch.float32


    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    seed_everything(cfg['system']['seed'])

    # Save config file
    model_dir = os.path.join(args.model_save_dir, 'cryoet_detection_model')
    os.makedirs(model_dir, exist_ok=True)
    config_save_path = os.path.join(model_dir, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f)
    print(f'Saved config to {config_save_path}')

    # generate copick_root object
    assert args.copick_config_path, "copick_config_path must be provided"
    copick_root = copick.from_file(args.copick_config_path)

    gt_sub_df = generate_gt_sub_df(
        copick_root=copick_root,
    )
    train_df = generate_sliding_window_index(
        copick_root=copick_root,
        image_size=cfg['dataset']['image_size'],
        image_stride=cfg['dataset']['image_stride'],
        voxel_spacing=cfg['dataset']['voxel_spacing'],
        include_edge_windows=False,
    )
    val_df = generate_sliding_window_index(
        copick_root=copick_root,
        image_size=cfg['dataset']['image_size'],
        image_stride=cfg['dataset']['image_stride'],
        voxel_spacing=cfg['dataset']['voxel_spacing'],
        include_edge_windows=True,
    )

    folds = split_by_run_name(
        df=train_df,
        n_folds=cfg['train_loop']['n_folds'],
        random_state=cfg['system']['seed'],
    )

    train_index = train_df.index.values[~train_df['experiment_id'].isin(folds[0])]
    val_index = val_df.index.values[val_df['experiment_id'].isin(folds[0])]

    train_dl, val_dl = build_dataloaders(
        copick_root=copick_root,
        df=train_df,
        val_df=val_df,
        train_index=train_index,
        val_index=val_index,
        cfg=cfg
    )
    
    train_single_fold(
        fold=0,
        train_dl=train_dl,
        val_dl=val_dl,
        val_df=val_df,
        val_index=val_index,
        val_sub_df=gt_sub_df[gt_sub_df['experiment'].isin(folds[0])],
        cfg=cfg,
        copick_root=copick_root,
        model_save_dir=args.model_save_dir
    )