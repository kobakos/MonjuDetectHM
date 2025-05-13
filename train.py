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

from src.utils.utils import load_configs
from src.component_factory import create_optimizer, create_scheduler, create_criterion, create_metric
from src.data_processing import build_dataloaders
from src.utils.loop import to_device
from src.utils import sigmoid
from src.models import build_model_from_config
from src.models.postprocessing import PostProcessor
from src.evaluation import score, pred_dicts_to_df

from src.losses import from_cfg

from src.logger import Logger, EmaCalculator

def train_single_fold(fold):
    print(f'============================================start of fold {fold}=============================================')
    # create model and whatnot
    model = build_model_from_config(cfg['model'])
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    optimizer = create_optimizer(cfg['optimizer'], model.parameters())
    scheduler = create_scheduler(cfg['scheduler'], optimizer)
    if log:
        logger.reset_best_metric()
    criterion = from_cfg(cfg['criterion'])
    if 'dann' in cfg['criterion']['name']:
        val_criterion = from_cfg(cfg['criterion']['classifier'])
    elif cfg['criterion']['name'] == 'bce_w_offset':
        criterion_cfg = cfg['criterion'].copy()
        criterion_cfg['name'] = 'bce'
        val_criterion = lambda x, y: from_cfg(criterion_cfg)(x[0], y)
    else:
        val_criterion = criterion
    metric = create_metric(cfg['metric'])
    post_processor = PostProcessor(
        classes = [
            'apo-ferritin',
            # 'beta-amylase', # ignore beta-amylase for now as it is deemed impossible to detect
            'beta-galactosidase',
            'ribosome',
            'thyroglobulin',
            'virus-like-particle'
        ],
        tiles_per_experiment = 162,
        **cfg["postprocessing"]
    )

    # data preparation
    val_df_fold = val_df[val_df['fold'] == fold]
    val_sub_df = train_sub_df[train_sub_df['experiment'].isin(val_df_fold['experiment_id'])]
    
    # main training loop
    loss_ema = EmaCalculator(cfg['system']['loss_alpha'])
    best_val_loss = 1e6
    for epoch in range(cfg['train_loop']['n_epochs']):
        train_dl, val_dl = build_dataloaders(train_df, val_df, fold, cfg, settings, pretrain=cfg['pretrain'], epoch=epoch)
        # training loop
        model.train()
        all_losses_mean = 0
        #with tqdm(train_dl, desc=f'Epoch: {epoch}, Training', total=len(train_dl), leave=False) as pbar:
        if args.tqdm:
            pbar = tqdm(train_dl, desc=f'Epoch: {epoch}, Training', total=len(train_dl), leave=False)
        else:
            pbar = train_dl
        for i, (images, targets) in enumerate(pbar):
            #print(images.shape)
            images, targets = to_device(images, targets, cfg['system']['device'])
            targets['epoch'] = epoch
            if cfg["model"].get("dann", False):
                images = images, epoch + i / len(train_dl)
            with torch.autocast(device_type = cfg['system']['device'], dtype=AMP_DTYPE):
                out = model(images)
                loss = criterion(out, targets)

            # update various loss metrics
            all_losses_mean = (all_losses_mean * i + loss.item()) / (i + 1)
            loss_ema.update(loss.item())
            if False and log and loss > 1.5 * loss_ema.ema:
                targets = targets['heatmap']
                t = targets.cpu().float().numpy() 
                if cfg["model"].get("dann", False):
                    o = out[0].cpu().detach().float().numpy()
                    im = images[0].cpu().numpy()
                elif cfg['criterion']['name'] == 'bce_w_offset':
                    o = out[0].cpu().detach().float().numpy()
                    im = images.cpu().numpy()
                else:
                    o = out.cpu().detach().float().numpy()
                    im = images.cpu().numpy()
                logger.log_heatmap_preds(
                    image=im,
                    target=t, 
                    output=o,
                    fold=fold,
                    epoch=epoch,
                    name='irregular_training_loss'
                )

            loss.backward()
            if cfg['train_loop']['max_grad_norm'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_loop']['max_grad_norm'])
            
            if (i + 1) % cfg['train_loop']['gradient_accumulation_steps'] == 0:
                if cfg['optimizer']['name'] == 'sam':
                    optimizer.first_step(zero_grad=True)
                    with torch.autocast(device_type = cfg['system']['device'], dtype=AMP_DTYPE):
                        out = model(images)
                        loss = criterion(out, targets)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if cfg['scheduler']['name'] == 'cosine_timm':
                scheduler.step(epoch + i / len(train_dl))
            else:
                scheduler.step()

            if i % cfg['system']['log_freq'] == 0 and log:
                logger.log({
                    'fold': fold,
                    'epoch': epoch,
                    'batch': i,
                    'train_loss': loss.item(),
                    'train_loss_ema': loss_ema.ema,
                    'train_step': epoch * len(train_dl) + i})      
            if args.tqdm:          
                pbar.set_postfix(OrderedDict(
                    loss = loss.item(),
                    loss_ema = loss_ema.ema,
                    loss_mean = all_losses_mean
                ))
            if args.sanity_check:
                break
        print(f'Epoch: {epoch}, Loss_ema: {loss_ema.ema: .5e}, Loss_mean: {all_losses_mean: .5e}')
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
            if cfg["model"].get("dann", False):
                images = images, epoch + i / len(train_dl)
    
            with torch.no_grad():
                with torch.autocast(device_type=cfg['system']['device'], dtype=INFER_AMP_DTYPE):
                    out = model(images)
                    #os.makedirs('./visualization_results/offset_map', exist_ok=True)
                    #for d0 in range(out[1][0].shape[1]):
                    #    m=out[1][0][:, d0].cpu().numpy().transpose((1, 2, 0))
                    #    print(m.min())
                    #    print(m.max())
                    #    m = m / 10
                    #    m += 0.5
                    #    m *= 255
                    #    m = m.clip(0, 255).astype(np.uint8)
                    #    cv2.imwrite(f'./visualization_results/offset_map/{d0}.png', m)
                    if cfg["model"].get("dann", False):
                        images = images[0]
                        out = out[0]
                    val_loss = val_criterion(out, targets)
            crop_origins = val_df_fold.iloc[i * cfg['train_loop']['val_batch_size']: (i + 1) * cfg['train_loop']['val_batch_size'], [2, 3, 4]].values
            experiment_ids = val_df_fold.iloc[i * cfg['train_loop']['val_batch_size']: (i + 1) * cfg['train_loop']['val_batch_size'], 0].values

            #if log and (i+1) % log_freq == 0:
            #    if cfg['criterion']['name'] == 'bce_w_offset':
            #        out = out[0]
            #    targets = targets['heatmap']
            #    t = targets.cpu().float().numpy() 
            #    o = out.cpu().float().numpy()
            #    im = images.cpu().numpy()
            #    logger.log_heatmap_preds(image=im, target=t, output=o, fold=fold, epoch=epoch)
            del images, targets
            post_processor.accumulate(out, crop_origins, experiment_ids)

            val_loss_mean = (val_loss_mean * i + val_loss.item()) / (i + 1)
            #exit()
        assert post_processor.accumulated_data == {}
        pred_sub_df = pred_dicts_to_df(post_processor.predictions)
        val_metric_mean = score(val_sub_df, pred_sub_df, None)#, distance_multiplier=1.0)
        if args.tqdm:
            pbar.close()
        best_val_loss = min(best_val_loss, val_loss_mean)
            
        print(f'Epoch: {epoch}, avg Validation Loss: {val_loss_mean: .5e}, avg Validation Metric: {val_metric_mean: .5f}')
        if log:
            values_to_log = {
                'fold': fold,
                'epoch': epoch,
                'avg_val_loss': val_loss_mean,
                'avg_val_metric': val_metric_mean,
                'avg_train_loss': all_losses_mean,
            }
            logger.log(values_to_log)
            save_by = cfg.get('save_by', 'val_metric')
            if save_by == 'val_metric':
                logger.update_best_metric(val_metric_mean, mode='max')
            elif save_by == 'val_loss':
                logger.update_best_metric(val_loss_mean, mode='min')
            else:
                raise ValueError(f'Unknown save_by: {save_by}')
            #if epoch == cfg['n_epochs'] - 1:
            logger.save_model(model, fold, epoch, best_only=True)
            if best_val_loss == val_loss_mean: 
                os.makedirs(f'../experiments/{logger.iteration_name}/weights/fold_{fold}', exist_ok=True)
                torch.save(model.state_dict(), f'../experiments/{logger.iteration_name}/weights/fold_{fold}/best_val_loss.pth')
            if logger.early_stop(patience=cfg['train_loop']['early_stop_patience']):
                print(f'Early stopping at epoch {epoch}')
                break
        torch.save(model.state_dict(), f'./latest.pth')
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
    parser.add_argument('--log', action='store_true', help='Enable logging')
    parser.add_argument('--tqdm', action='store_true', help='Use tqdm')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--sanity_check', action='store_true', help='Run sanity check with short steps/epochs')
    parser.add_argument('--settings', type=str, help='Path to settings.json', default='SETTINGS.json')
    args = parser.parse_args()

    # Load settings from SETTINGS.json
    settings = load_configs(args.settings)
    
    config_file = args.config
    with open(config_file, encoding='utf-8')as f:
        cfg = yaml.safe_load(f)

    # Make sure paths from settings are used in config
    BASE_PATH = Path(settings['base_path'])

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

    log = args.log
    if log:
        logger = Logger(cfg, BASE_PATH / settings["model_save_dir"], args.disable_wandb, metric_init=0 if cfg.get('save_by', 'val_metric') == 'val_metric' else 1e6)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    seed_everything(cfg['system']['seed'])

    processed_path = Path(settings['processed_data_path'])
    
    train_df = pd.read_csv(BASE_PATH / processed_path / cfg["paths"]["train_df_path"])
    val_df = pd.read_csv(BASE_PATH / processed_path / cfg["paths"]["val_df_path"])
    train_sub_df = pd.read_csv(BASE_PATH / processed_path / cfg["paths"]["train_sub_df_path"])

    log_freq = cfg['system']['log_freq']
    for fold in range(cfg['train_loop']['n_folds']):
        train_single_fold(fold)