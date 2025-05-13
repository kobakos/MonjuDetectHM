import os
import random
from math import ceil
from pathlib import Path

import zarr
import numpy as np
import pandas as pd

from src.data_processing.dataset import get_image, get_points, get_points_ext
from src.utils.utils import load_configs

def create_folds(fold_df_path, experiment_ids, n_folds = 4):
    if os.path.exists(fold_df_path):
        return pd.read_csv(fold_df_path)
    else:
        fold_df = {
            "experiment_id": [],
            "fold": [],
        }
        experiment_ids = list(experiment_ids)
        random.shuffle(experiment_ids)
        f = 0
        for experiment_id in experiment_ids:
            fold_df['experiment_id'].append(experiment_id)
            fold_df['fold'].append(f)
            f = (f + 1) % n_folds
        fold_df = pd.DataFrame(fold_df)
        fold_df.to_csv(fold_df_path, index=False)
        return fold_df
    
def get_windows(dim_len, crop_size, crop_stride, include_edge = True):
    n_crops = ceil((dim_len - crop_size) / crop_stride)
    crop_origins = [ii * crop_stride for ii in range(n_crops)] + ([dim_len - crop_size] if include_edge else [])
    return crop_origins

def create_dann_pretrain_df(save_path, settings):
    """
    This function will basically just connect pretrain_df.csv and
    train_df.csv (experimental only) zarr_path replaced to scaled_wbps
    """
    base_path = Path(settings['base_path'])
    processed_path = Path(settings['processed_path'])
    
    pretrain_df = pd.read_csv(base_path / processed_path / 'pretrain_df.csv')
    train_df = pd.read_csv(base_path / processed_path / 'train_df.csv')
    train_df = train_df[train_df['source'] == 'original']
    train_df["zarr_path"] = train_df["zarr_path"].apply(lambda x: f"{base_path}/{processed_path}/scaled_wbp/{x.split('/')[-1]}")

    dann_df = pd.concat([pretrain_df, train_df])
    dann_df.to_csv(save_path, index=False)

def create_train_df(
        settings,
        save_path = None,
        fold_df_path = None,
        crop_size:int = [128, 128, 128],
        resolution_hierarchy = 0,
        mode = 'train',
        do_not_validate_on_ext = True,
        include_edge = False,
    ):
    """
    This function will create a train_df.csv file that will be used to generate the dataset.
    This may create a new fold_df.csv file if it does not exist.
    """
    crop_stride = 64
    
    # Get paths from settings
    base_path = Path(settings['base_path'])
    competition_data_path = Path(settings['competition_data_path'])
    external_data_path = Path(settings['external_data_path']) 
    processed_path = Path(settings['processed_data_path'])
    
    # Set default paths based on settings if not provided
    if save_path is None:
        save_path = base_path / processed_path / f"{mode}_df.csv"
    if fold_df_path is None and mode == 'train':
        fold_df_path = base_path / processed_path / "fold_df.csv"

    if mode == 'train':
        experiment_ids = os.listdir(base_path / competition_data_path / 'train/static/ExperimentRuns')
        fold_df = create_folds(fold_df_path, experiment_ids)
        fold_df.set_index('experiment_id', inplace=True)

    train_df = {
        "experiment_id": [],
        "zarr_path": [],
        "crop_origin_d0": [],
        "crop_origin_d1": [],
        "crop_origin_d2": [],
        "source": [],
    }
    if mode=='train':
        train_df['fold'] = []
    
    for experiment_id in os.listdir(base_path / competition_data_path / mode / 'static' / 'ExperimentRuns'):
        zarr_path = base_path / competition_data_path / mode / 'static' / 'ExperimentRuns' / experiment_id / 'VoxelSpacing10.000' / 'denoised.zarr'
        assert os.path.exists(zarr_path)
        arr = zarr.open(str(zarr_path))[resolution_hierarchy]
        assert arr.shape == (184, 630, 630)
        arr_shape = arr.shape
        
        n_crops_0 = ceil((arr_shape[0] - crop_size[0]) / crop_stride) + 1
        n_crops_1 = ceil((arr_shape[1] - crop_size[1]) / crop_stride) + 1
        n_crops_2 = ceil((arr_shape[2] - crop_size[2]) / crop_stride) + 1
        crop_origins_d0 = [ii * crop_stride for ii in range(n_crops_0-1)] + [arr_shape[0] - crop_size[0]]
        crop_origins_d1 = [ii * crop_stride for ii in range(n_crops_1-1)] + [arr_shape[1] - crop_size[1]]
        crop_origins_d2 = [ii * crop_stride for ii in range(n_crops_2-1)] + [arr_shape[2] - crop_size[2]]
        for i in range(n_crops_0):
            crop_start_0 = crop_origins_d0[i]
            for j in range(n_crops_1):
                crop_start_1 = crop_origins_d1[j]
                for k in range(n_crops_2):
                    # kind of a junky fix to avoid including the edge crops
                    if not include_edge and (i == n_crops_0 - 1 or j == n_crops_1 - 1 or k == n_crops_2 - 1):
                        continue

                    crop_start_2 = crop_origins_d2[k]
                    train_df['experiment_id'].append(experiment_id)
                    train_df['zarr_path'].append(str(zarr_path))
                    train_df['crop_origin_d0'].append(crop_start_0)
                    train_df['crop_origin_d1'].append(crop_start_1)
                    train_df['crop_origin_d2'].append(crop_start_2)
                    train_df['source'].append('original')
                    if mode=='train':
                        train_df['fold'].append(fold_df.loc[experiment_id, 'fold'])
                        
    if mode == 'test':
        train_df = pd.DataFrame(train_df)
        train_df.to_csv(save_path, index=False)
        return
    
    crop_stride = 128
    for ext_experiment_id in os.listdir(base_path / external_data_path / 'Simulated'):
        zarr_path = base_path / external_data_path / 'Simulated' / ext_experiment_id / 'Reconstructions' / 'VoxelSpacing10.000' / 'Tomograms' / '100' / f'{ext_experiment_id}.zarr'
        assert os.path.exists(zarr_path), zarr_path
        arr = zarr.open(str(zarr_path))[resolution_hierarchy]
        assert arr.shape == (200, 630, 630), arr.shape
        arr_shape = (200, 630, 630)

        n_crops_0 = ceil((arr_shape[0]-crop_size[0]) / crop_stride) + 1
        n_crops_1 = ceil((arr_shape[1]-crop_size[1]) / crop_stride) + 1
        n_crops_2 = ceil((arr_shape[2]-crop_size[2]) / crop_stride) + 1
        crop_origins_d0 = [ii * crop_stride for ii in range(n_crops_0-1)] + [arr_shape[0] - crop_size[0]]
        crop_origins_d1 = [ii * crop_stride for ii in range(n_crops_1-1)] + [arr_shape[1] - crop_size[1]]
        crop_origins_d2 = [ii * crop_stride for ii in range(n_crops_2-1)] + [arr_shape[2] - crop_size[2]]
        for i in range(n_crops_0):
            crop_start_0 = crop_origins_d0[i]
            for j in range(n_crops_1):
                crop_start_1 = crop_origins_d1[j]
                for k in range(n_crops_2):
                    crop_start_2 = crop_origins_d2[k]
                    train_df['experiment_id'].append(ext_experiment_id)
                    train_df['zarr_path'].append(str(zarr_path))
                    train_df['crop_origin_d0'].append(crop_start_0)
                    train_df['crop_origin_d1'].append(crop_start_1)
                    train_df['crop_origin_d2'].append(crop_start_2)
                    train_df['source'].append('external')
                    if do_not_validate_on_ext:
                        train_df['fold'].append(-1)
                    else:
                        train_df['fold'].append(fold_df.loc[ext_experiment_id, 'fold'])
    train_df = pd.DataFrame(train_df)
    train_df.to_csv(save_path, index=False)

def create_train_sub_df(
        settings,
        save_path = None,
        train_df_path = None,
    ):
    """
    generates a dataframe with the same structure as the submission dataframe, which can be used for validation
    """
    base_path = Path(settings['base_path'])
    competition_data_path = Path(settings['competition_data_path'])
    processed_path = Path(settings['processed_data_path'])
    
    # Set default paths based on settings if not provided
    if save_path is None:
        save_path = base_path / processed_path / "train_sub_df.csv"
    if train_df_path is None:
        train_df_path = base_path / processed_path / "train_df.csv"
    
    train_df = pd.read_csv(train_df_path)
    train_sub_df = {
        "id": [],
        "experiment": [],
        "particle_type": [],
        "x": [],
        "y": [],
        "z": [],
    }
    experiment_ids = train_df.loc[train_df["source"] == "original", 'experiment_id'].unique()
    for experiment_id in experiment_ids:
        for cls in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle', 'beta-amylase']:
            points = get_points(experiment_id, base_path / competition_data_path / f'train/overlay/ExperimentRuns/{experiment_id}/Picks/{cls}.json')
            for point in points:
                train_sub_df['id'].append(len(train_sub_df['id']))
                train_sub_df['experiment'].append(experiment_id)
                train_sub_df['particle_type'].append(cls)
                train_sub_df['x'].append(point[2])
                train_sub_df['y'].append(point[1])
                train_sub_df['z'].append(point[0])
    train_sub_df = pd.DataFrame(train_sub_df)
    train_sub_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    # Load settings from SETTINGS.json
    settings = load_configs('SETTINGS.json')
    
    # Create train dataframe
    create_train_df(
        settings=settings,
        save_path=Path(settings['base_path']) / Path(settings['processed_data_path']) / 'train_df.csv'
    )
    create_train_df(
        settings=settings,
        save_path=Path(settings['base_path']) / Path(settings['processed_data_path']) / 'val_df.csv',
        include_edge = True,
    )
    
    # Create submission dataframe
    create_train_sub_df(
        settings=settings,
        save_path=Path(settings['base_path']) / Path(settings['processed_data_path']) / 'train_sub_df.csv'
    )