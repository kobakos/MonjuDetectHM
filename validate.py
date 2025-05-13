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
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from src.component_factory import create_optimizer, create_scheduler, create_criterion, create_metric
from src.data_processing import build_dataloaders
from src.data_processing.dataset import dataset
from src.utils.loop import to_device
from src.utils import sigmoid
from src.models import build_model_from_config
from src.models.postprocessing import PostProcessor, weighted_box_fusion
from src.evaluation import score, pred_dicts_to_df
from visualization import plot_predictions


from src.utils import sigmoid

import src.constants as constants

from src.losses import from_cfg

from src.logger import Logger, EmaCalculator

from src import constants
from src.utils import load_configs

settings = load_configs("SETTINGS.json")
BASE_PATH = Path(settings['base_path'])

# set random seed for everything
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#experiment_names = ['46_resnet50d.ra2_in1k_20241215']

experiment_names = [
    "1_resnet50d.ra2_in1k_20250513"
]  

OFFSET = False

model_soup=False

weight_path_folds = [
    [[BASE_PATH / settings["model_save_dir"] / e / "weights" / f"fold_{i}" / 'best.pth'
      for e in experiment_names]] for i in range(4)
]
model_config_path_folds = [
    [[BASE_PATH / settings["model_save_dir"] / e / 'config.yml'
      for e in experiment_names]]
    for i in range(4)
]

import argparse
parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--config', type=str, help='Path to the config file (in json)', default='configs/infer_config.yml')
parser.add_argument('--detect_anomaly', action='store_true', help='Enable anomaly detection')
parser.add_argument('--log', action='store_true', help='Enable logging')
parser.add_argument('--tqdm', action='store_true', help='Use tqdm')
parser.add_argument('--show', action='store_true', help='Save results')
parser.add_argument('--more-tolerant', action='store_true', help='Use distance multiplier 1.0 instead of 0.5')
parser.add_argument('--disable-wandb', action='store_true', help='Disable wandb')
args = parser.parse_args()

config_file = args.config
with open(config_file, encoding='utf-8')as f:
    cfg = yaml.safe_load(f)

if cfg['system']['amp_dtype'] == 'fp16':
    AMP_DTYPE = torch.float16
elif cfg['system']['amp_dtype'] == 'bf16':
    AMP_DTYPE = torch.bfloat16
elif cfg['system']['amp_dtype'] == 'fp32':
    AMP_DTYPE = torch.float32

if cfg['system']['infer_amp_dtype'] == 'fp16':
    INFER_AMP_DTYPE = torch.float16
elif cfg['system']['infer_amp_dtype'] == 'bf16':
    INFER_AMP_DTYPE = torch.bfloat16
elif cfg['system']['infer_amp_dtype'] == 'fp32':
    INFER_AMP_DTYPE = torch.float32

log = args.log
if log:
    logger = Logger(cfg, args.disable_wandb, metric_init=1e9)

if args.detect_anomaly:
    torch.autograd.set_detect_anomaly(True)
seed_everything(cfg['system']['seed'])

train_df = pd.read_csv(BASE_PATH / settings["processed_data_path"] / cfg['paths']['train_df_path'])
val_df = pd.read_csv(BASE_PATH / settings["processed_data_path"] / cfg['paths']['val_df_path'])
train_sub_df = pd.read_csv(BASE_PATH / settings["processed_data_path"] / cfg['paths']['train_sub_df_path'])


#val_df.reset_index(drop=True, inplace=True)
#val_df['crop_origin_d0'] = 4
#print(val_df)
val_df = val_df[val_df['crop_origin_d0'] == 0]
#val_df.loc[:, 'crop_origin_d0'] = 12
cfg['feature_extractor_args']['image_size'] = (192, 128, 128)
vis_dir = experiment_names[0] + 'TTA'



Transformations=[
    'e', # no transformation
    'b', #flip1
    'ba2', # flip2
    'a2', # flip12,
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


pseudo_label_epochs = 0

TEST = False
ADD_FP = False#True
DOUBLE_PRED = False
PARTICLES_TO_TEST = 'beta-galactosidase'


import zarr
from math import ceil
def create_infer_df(
        save_path,
        base_path,
        crop_size = (128, 128, 128),
        resolution_hierarchy = 0,
    ):
    """
    This function will create a train_df.csv file that will be used to generate the dataset.
    This may create a new fold_df.csv file if it does not exist.
    """
    # Ensure base_path is a Path object
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    # Default base_path to project data directory if unset
    if base_path is None:
        base_path = BASE_PATH / 'data' / 'original'
    
    #fold_df = create_folds()
    fold_df = {
        "TS_99_9": 0,
        "TS_6_4": 0,
        "TS_86_3": 1,
        "TS_6_6": 1,
        "TS_69_2": 2,
        "TS_73_6": 2,
        "TS_5_4": 3,
    }
    #fold_df.set_index('experiment_id', inplace=True)

    train_df = {
        "experiment_id": [],
        "zarr_path": [],
        "crop_origin_d0": [],
        "crop_origin_d1": [],
        "crop_origin_d2": [],
        "fold": [],
        "source": [],
    }
    
    crop_stride=64
    for experiment_id in os.listdir(base_path / 'train' / 'static' / 'ExperimentRuns'):
        zarr_path = base_path/'train'/'static'/'ExperimentRuns'/experiment_id/'VoxelSpacing10.000'/'denoised.zarr'# only used 'denoised.zarr' as only this is available on test data
        assert os.path.exists(zarr_path)# opening a non-existent zarr file will create it (wtf???), so we need to check if it exists
        arr = zarr.open(str(zarr_path))[resolution_hierarchy]
        assert arr.shape == (184, 630, 630)# Shape of the train data is assumed to be (184, 630, 630)
        arr_shape = arr.shape
        
        n_crops_0 = ceil((arr_shape[0] - crop_size[0]) / crop_stride)
        n_crops_1 = ceil((arr_shape[1] - crop_size[1]) / crop_stride) + 1
        n_crops_2 = ceil((arr_shape[2] - crop_size[2]) / crop_stride) + 1
        crop_origins_d0 = [ii * crop_stride for ii in range(n_crops_0)] #+ [arr_shape[0] - crop_size[0]]
        crop_origins_d1 = [ii * crop_stride for ii in range(n_crops_1-1)] + [arr_shape[1] - crop_size[1]]
        crop_origins_d2 = [ii * crop_stride for ii in range(n_crops_2-1)] + [arr_shape[2] - crop_size[2]]
        for i in range(n_crops_0):
            crop_start_0 = crop_origins_d0[i]
            for j in range(n_crops_1):
                crop_start_1 = crop_origins_d1[j]
                for k in range(n_crops_2):
                    crop_start_2 = crop_origins_d2[k]
                    train_df['experiment_id'].append(experiment_id)
                    train_df['zarr_path'].append(str(zarr_path))
                    train_df['crop_origin_d0'].append(crop_start_0)
                    train_df['crop_origin_d1'].append(crop_start_1)
                    train_df['crop_origin_d2'].append(crop_start_2)
                    train_df['fold'].append(fold_df[experiment_id])
                    train_df['source'].append('original')
    train_df = pd.DataFrame(train_df)
    return train_df

#val_df = create_infer_df(save_path = '../data/processed/infer_df.csv', base_path = '../data/original', crop_size=(160, 128, 128))
#pseudo_label_epochs = 1

def collate_fn(batch):
    out = tuple(zip(*batch))
    images, targets = out
    images = torch.stack(images)
    return images, targets
def build_infer_loader(infer_df, cfg):
    infer_ds = dataset(
        index = infer_df.index.values,
        df = infer_df,
        image_path = cfg['data_path'] + '/test',
        return_targets=False,
        **cfg['feature_extractor_args']
    )
    infer_dl = torch.utils.data.DataLoader(
        infer_ds,
        batch_size = cfg['val_batch_size'],
        shuffle = False,
        num_workers = 4,
        prefetch_factor = 2,
        collate_fn = collate_fn,
        pin_memory = True
    )
    return infer_dl
import copy
def train_on_pseudo(model, infer_df, cfg, pseudo_label_epochs):
    model_teacher = copy.deepcopy(model).eval()
    cfg = copy.deepcopy(cfg)
    #model_teacher = torch.compile(model_teacher)
    model.train()
    pseudo_df = infer_df.copy()
    #pseudo_df = pseudo_df[(pseudo_df['crop_origin_d0'].values == 0) & (pseudo_df['crop_origin_d1'].values == 320) & (pseudo_df['crop_origin_d2'].values == 320)]
    pseudo_df = pseudo_df.sample(frac=1, random_state=3045)
    
    cfg["feature_extractor_args"]["image_size"] = [128, 128, 128]
    cfg["val_batch_size"] = 2
    cfg['optimizer']['lr'] = 1e-5
    _, pseudo_dl = build_dataloaders(train_df, pseudo_df, fold, cfg, pretrain=False)
    #pseudo_dl = build_infer_loader(pseudo_df, cfg)
    assert len(models) == 1, "pseudo labelling should only be done with 1 model"
    
    optimizer = create_optimizer(cfg['optimizer'], model.parameters())
    #scheduler = create_scheduler(cfg['scheduler'], optimizer)
    #criterion = from_cfg(cfg['criterion']) 
    pos_weight = torch.Tensor([20.0, 16.0, 18.0, 16.0, 20.0]).to('cuda')[None, :, None, None, None]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(pseudo_label_epochs):
        for i, (images, targets) in enumerate(pseudo_dl):
            images = images.to('cuda')
            targets = targets.to('cuda')
            #print(images.shape)
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                with torch.no_grad():
                    #plabel = torch.nn.functional.sigmoid(model_teacher(images))
                    plabel = model_teacher(images)
                    plabel += model_teacher(images.flip((-1,))).flip((-1,))
                    plabel += model_teacher(images.flip((-2,))).flip((-2,))
                    plabel += model_teacher(images.flip((-2,-1))).flip((-2,-1))
            plabel /= 4
    
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, torch.nn.functional.sigmoid(plabel).float())
            loss.backward()
            if cfg['train_loop']['max_grad_norm'] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train_loop']['max_grad_norm'])
            
            if cfg['train_loop']['gradient_accumulation_steps'] > 1 and (i + 1) % cfg['train_loop']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
                
            #if cfg['scheduler']['name'] == 'cosine_timm':
            #    scheduler.step(epoch + i / len(pseudo_dl))
            #else:
            #    scheduler.step()
    return model

def infer_transformerd(model, images, transformation):
    if transformation == 'e':
        out = model(images)
    elif transformation == 'b':
        out = model(images.flip(dims=(-2,))).flip(dims=(-2,))
    elif transformation == 'ba2':
        out = model(images.flip(dims=(-1,))).flip(dims=(-1,))
    elif transformation == 'a2':
        out = model(images.flip(dims=(-1, -2))).flip(dims=(-1, -2))
    elif transformation == 'a':
        out = model(images.rot90(dims=(-1, -2), k=1)).rot90(dims=(-1, -2), k=3)
    elif transformation == 'a3':
        out = model(images.rot90(dims=(-1, -2), k=3)).rot90(dims=(-1, -2), k=1)
    elif transformation == 'ba':
        out = model(images.flip(dims=(-2,)).rot90(dims=(-1, -2), k=1)).rot90(dims=(-1, -2), k=3).flip(dims=(-2,))
    elif transformation == 'ba3':
        out = model(images.flip(dims=(-2,)).rot90(dims=(-1, -2), k=3)).rot90(dims=(-1, -2), k=1).flip(dims=(-2,))
    elif transformation == 'c':
        out = model(images.flip(dims=(-3,))).flip(dims=(-3,))
    elif transformation == 'cb':
        out = model(images.flip(dims=(-3, -2))).flip(dims=(-3, -2))
    elif transformation == 'cba2':
        out = model(images.flip(dims=(-3, -1))).flip(dims=(-3, -1))
    elif transformation == 'ca2':
        out = model(images.flip(dims=(-3, -2, -1))).flip(dims=(-3, -2, -1))
    elif transformation == 'ca':
        out = model(images.flip(dims=(-3,)).rot90(dims=(-2, -1), k=1)).rot90(dims=(-2, -1), k=3).flip(dims=(-3,))
    elif transformation == 'ca3':
        out = model(images.flip(dims=(-3,)).rot90(dims=(-2, -1), k=3)).rot90(dims=(-2, -1), k=1).flip(dims=(-3,))
    elif transformation == 'cba':
        out = model(images.flip(dims=(-3, -2)).rot90(dims=(-1, -2), k=1)).rot90(dims=(-1, -2), k=3).flip(dims=(-3, -2))
    elif transformation == 'cba3':
        out = model(images.flip(dims=(-3, -2)).rot90(dims=(-1, -2), k=3)).rot90(dims=(-1, -2), k=1).flip(dims=(-3, -2))
    else:
        raise ValueError(f'Unknown transformation: {transformation}')
    
    #if isinstance(out, tuple):
    #    out = out[0]
    if out.shape[2] > 184:
        out = out[:, :, :184, :, :]
    return out

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
        if out[0].shape[2] > 184:
            out = (
                out[0][:, :, :184, :, :],
                out[1][:, :, :184, :, :]
            )
    else:
        out = transformations[reverse_transformations[transformation]](out)
        if out.shape[2] > 184:
            out = out[:, :, :184, :, :]
    
    return out

def infer(models, images, TTA):
    out = 0
    with torch.inference_mode():
        for model in models:
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                for t in TTA:
                    out += infer_transformerd(model, images, t)
        out /= len(models)
        out /= len(TTA)
    return out

def infer_offset(models, images, TTA):
    out = [0, 0]
    with torch.inference_mode():
        for model in models:
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                for t in TTA:
                    out_tup = infer_transformerd(model, images, t)
                    out[0] += out_tup[0]
                    out[1] += out_tup[1]
        out[0] /= len(models)
        out[1] /= len(models)
        out[0] /= len(TTA)
        out[1] /= len(TTA)
    return tuple(out)

def TTA_avg(models, pbar, TTA, cfg, AMP_DTYPE):
    postprocess_size = list(cfg['feature_extractor_args']['image_size'])
    postprocess_size[0] = min(184, postprocess_size[0])
    post_processor = PostProcessor(
        classes = [
            'apo-ferritin',
            # 'beta-amylase', # ignore beta-amylase for now as it is deemed impossible to detect
            'beta-galactosidase',
            'ribosome',
            'thyroglobulin',
            'virus-like-particle'
        ],
        #tiles_per_experiment = 162,
        tiles_per_experiment = (val_df["experiment_id"] == val_df["experiment_id"][0]).sum(),
        window_size=postprocess_size,
        **cfg['postprocessing'],
        ignore_uncovered=True,
        keep_heatmaps=True,
        keep_accumulated=True
    )
    time_postrocess = 0
    time_inference = 0
    for i, (images, targets) in enumerate(pbar):
        #print(images.shape)
        images, targets = to_device(images, targets, cfg['system']['device'])

        start_inference = time.time()
        if OFFSET:
            out = infer_offset(models, images, TTA)
        else:
            out = infer(models, images, TTA)
        #out = out.sigmoid()
        #out = torch.nn.functional.sigmoid(out)
        time_inference += time.time() - start_inference

        start_postprocess = time.time()
        crop_origins = val_df_fold.iloc[i * cfg['train_loop']['val_batch_size']: (i + 1) * cfg['train_loop']['val_batch_size'], [2, 3, 4]].values
        experiment_ids = val_df_fold.iloc[i * cfg['train_loop']['val_batch_size']: (i + 1) * cfg['train_loop']['val_batch_size'], 0].values
        post_processor.accumulate(out, crop_origins, experiment_ids)
        time_postrocess += time.time() - start_postprocess
    #assert post_processor.accumulated_data == {}
    end = time.time()
    return post_processor, start-end, time_postrocess, time_inference

from src.models.postprocessing import weighted_box_fusion
def TTA_wbf_model(models, pbar, TTA, cfg, AMP_DTYPE):
    postprocess_size = list(cfg['feature_extractor_args']['image_size'])
    postprocess_size[0] = min(184, postprocess_size[0])
    post_processors = [
        PostProcessor(
            classes = [
                'apo-ferritin',
                # 'beta-amylase', # ignore beta-amylase for now as it is deemed impossible to detect
                'beta-galactosidase',
                'ribosome',
                'thyroglobulin',
                'virus-like-particle'
            ],
            #tiles_per_experiment = 162,
            tiles_per_experiment = (val_df["experiment_id"] == val_df["experiment_id"][0]).sum(),
            window_size=postprocess_size,
            **cfg['postprocessing'],
            ignore_uncovered=True,
            keep_heatmaps=True
        ) for _ in range(len(models))
    ]
    time_postrocess = 0
    time_inference = 0
    start = time.time()
    for i, (images, targets) in enumerate(pbar):
        #print(images.shape)
        images, targets = to_device(images, targets, cfg['system']['device'])

        start_inference = time.time()
        outs = [0 for _ in range(len(models))]
        with torch.inference_mode():
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                for mi, model in enumerate(models):
                    #outs[mi] = torch.zeros((images.shape[0], 5, 184, 128, 128), dtype=torch.float, device=cfg['system']['device'])
                    #continue
                    if OFFSET:
                        out = infer_offset(model, images, TTA)
                    else:
                        out = infer([model], images, TTA)
                    outs[mi] = out
        time_inference += time.time() - start_inference

        start_postprocess = time.time()
        crop_origins = val_df_fold.iloc[i * cfg['val_batch_size']: (i + 1) * cfg['val_batch_size'], [2, 3, 4]].values
        experiment_ids = val_df_fold.iloc[i * cfg['val_batch_size']: (i + 1) * cfg['val_batch_size'], 0].values
        for out, post_processor in zip(outs, post_processors, strict=True):
            post_processor.accumulate(out, crop_origins, experiment_ids)
        time_postrocess += time.time() - start_postprocess
    
    preds = {}
    # exp_id -> class -> points
    #       |-> heatmap
    all_predictions = [pp.predictions for pp in post_processors]
    for exp_id in all_predictions[0]:
        preds[exp_id] = {
            'heatmap': sum([pp[exp_id]['heatmap'] for pp in all_predictions]) / len(all_predictions)
        }
        for c in constants.classes:
            if c == 'beta-amylase':
                continue
            preds[exp_id][c] = {
                'points': np.vstack([pp[exp_id][c]['points'] for pp in all_predictions]),
                'confidence': np.concatenate([pp[exp_id][c]['confidence'] for pp in all_predictions])
            }
    for exp_id in preds:
        for c in preds[exp_id]:
            if c == 'heatmap':
                continue
            if len(preds[exp_id][c]['points']) == 0:
                continue
            preds[exp_id][c] = weighted_box_fusion(
                preds[exp_id][c],
                constants.particle_radius[c] * cfg['ensemble']['radius_multiplier'],
                cfg['ensemble']['min_votes'],
            )

    end = time.time()
        
    return preds, start-end, time_postrocess, time_inference

from src.models.postprocessing import weighted_box_fusion
def TTA_wbf(models, pbar, TTA, cfg, AMP_DTYPE):
    post_processors = [
        PostProcessor(
            classes = [
                'apo-ferritin',
                # 'beta-amylase', # ignore beta-amylase for now as it is deemed impossible to detect
                'beta-galactosidase',
                'ribosome',
                'thyroglobulin',
                'virus-like-particle'
            ],
            #tiles_per_experiment = 162,
            tiles_per_experiment = (val_df["experiment_id"] == val_df["experiment_id"][0]).sum(),
            window_size=cfg['feature_extractor_args']['image_size'],
            **cfg['postprocessing'],
            ignore_uncovered=True,
            keep_heatmaps=True
        ) for _ in range(len(TTA))
    ]
    time_postrocess = 0
    time_inference = 0
    start = time.time()
    for i, (images, targets) in enumerate(pbar):
        #print(images.shape)
        images, targets = to_device(images, targets, cfg['system']['device'])

        start_inference = time.time()
        outs = [0 for _ in range(len(TTA))]
        with torch.inference_mode():
            with torch.autocast(device_type=cfg['system']['device'], dtype=AMP_DTYPE):
                for ti, t in enumerate(TTA):
                    for mi, model in models:
                        outs[ti] += infer_transformerd(model, images, t)
                    outs[ti] /= len(models)
        time_inference += time.time() - start_inference

        start_postprocess = time.time()
        crop_origins = val_df_fold.iloc[i * cfg['val_batch_size']: (i + 1) * cfg['val_batch_size'], [2, 3, 4]].values
        experiment_ids = val_df_fold.iloc[i * cfg['val_batch_size']: (i + 1) * cfg['val_batch_size'], 0].values
        for out, post_processor in zip(outs, post_processors, strict=True):
            post_processor.accumulate(out, crop_origins, experiment_ids)
        time_postrocess += time.time() - start_postprocess
    
    preds = {}
    # exp_id -> class -> points
    #       |-> heatmap
    all_predictions = [pp.predictions for pp in post_processors]
    for exp_id in all_predictions[0]:
        preds[exp_id] = {
            'heatmap': sum([pp[exp_id]['heatmap'] for pp in all_predictions]) / len(all_predictions)
        }
        for c in constants.classes:
            if c == 'beta-amylase':
                continue
            preds[exp_id][c] = {
                'points': np.vstack([pp[exp_id][c]['points'] for pp in all_predictions]),
                'confidence': np.concatenate([pp[exp_id][c]['confidence'] for pp in all_predictions])
            }
    for exp_id in preds:
        for c in preds[exp_id]:
            if c == 'heatmap':
                continue
            preds[exp_id][c] = weighted_box_fusion(
                preds[exp_id][c],
                constants.particle_radius[c] * cfg['ensemble']['radius_multiplier'],
                cfg['ensemble']['min_votes'],
            )

    end = time.time()
        
    return preds, start-end, time_postrocess, time_inference

#from sklearn.cluster import DBSCAN
total_scores = []
torch.cuda.empty_cache()
combined_pred_df = pd.DataFrame()

scores = {}# fold: (threshold, score)

for fold in range(cfg['train_loop']['n_folds']):
    start = time.time()
    time_postrocess = 0
    time_inference = 0
    print(f'======================start of fold {fold}=======================')

    # create model and whatnot

    models = []
    weight_paths = weight_path_folds[fold]
    model_config_paths = model_config_path_folds[fold]
    for ps, mcfgs in zip(weight_paths, model_config_paths):
        models_i = []
        for p, mcfg in zip(ps, mcfgs):
            with open(mcfg, encoding='utf-8') as f:
                model_cfg = yaml.safe_load(f)["model"]
            if 'ema' in model_cfg:
                del model_cfg['ema']
            model_cfg['pretrained'] = False
            #model_cfg['w_offset_head'] = False
            if 'pretrain_weight_path' in model_cfg:
                del model_cfg['pretrain_weight_path']
            model = build_model_from_config(model_cfg)
            #model = torch.compile(model)
            state_dict = torch.load(p, map_location=cfg['system']['device'])
            if '_orig_mod.segmentation_head.weight' in state_dict:
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            model.eval()
            models_i.append(model)
        if not model_soup:
            models = models_i
            break
        model = models_i[0]
        for k in model.state_dict():
            model.state_dict()[k] = sum([m.state_dict()[k] for m in models_i])/len(models_i)
        models.append(model)

    

    if pseudo_label_epochs > 0:
        assert len(models) == 1, "pseudo labelling should only be done with 1 model"
        model = train_on_pseudo(models[0], val_df, cfg, pseudo_label_epochs)
        models = [model]

    # data preparation
    train_dl, val_dl = build_dataloaders(train_df, val_df, fold, cfg, settings, pretrain=cfg['pretrain'])
    val_df_fold = val_df[val_df['fold'] == fold]
    val_sub_df = train_sub_df[train_sub_df['experiment'].isin(val_df_fold['experiment_id'])]
    
    val_loss_mean = 0
    val_metric_mean = 0
    if args.tqdm:
        pbar = tqdm(val_dl, desc=f'Validation', total=len(val_dl), leave=False)
    else:
        pbar = val_dl
    
    if cfg['ensemble']['method'] == 'avg':
        post_processor, total_time, total_time_inference, total_time_postprocess = TTA_avg(models, pbar, Transformations, cfg, AMP_DTYPE)
        preds = post_processor.predictions
        #fold_scores = []
        #thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #for th in thresholds:
        #    preds = {}
        #    for exp_id in post_processor.predictions:
        #        preds[exp_id] = post_processor.process(torch.tensor(post_processor.predictions[exp_id]['heatmap']).cuda(), threshold=th)
        #        print(f'score for {exp_id}')
        #        print(score(val_sub_df[val_sub_df['experiment'] == exp_id], pred_dicts_to_df({exp_id: preds[exp_id]}), distance_multiplier=1.0 if args.more_tolerant else 0.5))
            #pred_sub_df = pred_dicts_to_df(preds)
            ##combined_pred_df = pd.concat([combined_pred_df, pred_sub_df])
            #val_metric_mean = score(val_sub_df, pred_sub_df, distance_multiplier=1.0 if args.more_tolerant else 0.5)
            #fold_scores.append(val_metric_mean)
            #print(f'avg Validation Metric at threshold {th}: {val_metric_mean: .5f}\n')
            #total_scores.append(val_metric_mean)
        #scores[fold] = (thresholds, fold_scores)
        #preds = post_processor.predictions
    elif cfg['ensemble']['method'] == 'wbf':
        preds, total_time, total_time_inference, total_time_postprocess = TTA_wbf(models, pbar, Transformations, cfg, AMP_DTYPE)
    elif cfg['ensemble']['method'] == 'wbf_model':
        preds, total_time, total_time_inference, total_time_postprocess = TTA_wbf_model(models, pbar, Transformations, cfg, AMP_DTYPE)
    else:
        raise ValueError(f'Unknown ensemble method: {cfg["ensemble"]["method"]}')

    
    dummy_FP = np.array([
        [   0,    0,    0],
        [   0,    0, 1840],
        [   0, 6300,    0],
        [   0, 6300, 1840],
        [6300,    0,    0],
        [6300,    0, 1840],
        [6300, 6300,    0],
        [6300, 6300, 1840],
    ])
    if ADD_FP:
        for exp_id in preds:
            preds[exp_id][PARTICLES_TO_TEST]['points'] = np.vstack([
                preds[exp_id][PARTICLES_TO_TEST]['points'],
                dummy_FP
            ]) if len(preds[exp_id][PARTICLES_TO_TEST]['points']) > 0 else dummy_FP
    if DOUBLE_PRED:
        for exp_id in preds:
            preds[exp_id][PARTICLES_TO_TEST]['points'] = np.vstack([
                preds[exp_id][PARTICLES_TO_TEST]['points'],
                preds[exp_id][PARTICLES_TO_TEST]['points']
            ]) if len(preds[exp_id][PARTICLES_TO_TEST]['points']) > 0 else preds[exp_id][PARTICLES_TO_TEST]['points']
    pred_sub_df = pred_dicts_to_df(preds)
    if TEST:
        pred_sub_df = pred_sub_df[pred_sub_df['particle_type'] == PARTICLES_TO_TEST]
    combined_pred_df = pd.concat([combined_pred_df, pred_sub_df])
    val_metric_mean = score(val_sub_df, pred_sub_df, distance_multiplier=1.0 if args.more_tolerant else 0.5)
    total_scores.append(val_metric_mean)
    print(f'avg Validation Loss: {val_loss_mean: .5e}, avg Validation Metric: {val_metric_mean: .5f}')
    print()
    del models
    torch.cuda.empty_cache()


    if args.show:
        for exp_id in preds:
            hmap = preds[exp_id]['heatmap']
            hmap = sigmoid(hmap)
            os.makedirs(f'visualization_results/{vis_dir}/{exp_id}', exist_ok=True)
    
            im = np.asarray(zarr.open(val_df[val_df['experiment_id'] == exp_id]['zarr_path'].values[0])[0])
            im = im*1e5
            im = im / 6 + 0.5
            im = im.clip(0, 1)
            im = (im * 255).astype(np.uint8)
    
            target = {}
            exp_df = val_sub_df[val_sub_df['experiment'] == exp_id]
            for c in constants.classes:
                target[c] = exp_df[exp_df['particle_type'] == c][['z', 'y', 'x']].values / 10
    
            points_to_vis = {}
            for c in constants.classes:
                if c in preds[exp_id]:
                    points_to_vis[c] = preds[exp_id][c]['points'][:,::-1] / 10
            plot_predictions(f'visualization_results/{vis_dir}/{exp_id}/', im, hmap, points_to_vis, target)
combined_score = score(train_sub_df, combined_pred_df, distance_multiplier=1.0 if args.more_tolerant else 0.5)
print(f'Combined score: {combined_score:.5f}')
for fold in range(cfg['n_folds']):
    plt.plot(scores[fold][0], scores[fold][1], label=f'fold {fold}')
plt.legend()
os.makedirs(f'visualization_results/{experiment_names[0]}', exist_ok=True)
plt.savefig(f'visualization_results/{experiment_names[0]}/threshold_scores.png')
