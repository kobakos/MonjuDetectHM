import os
import json
import random
from math import ceil
from pathlib import Path
from functools import partial
from typing import Tuple, Union, List, Optional

import cv2
import time
import zarr
import torch
import ndjson
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import copick
from copick.impl.filesystem import CopickRootFSSpec

#from models.detr.util.misc import NestedTensor
from .augmentations import Augmentator, generate_affine_mats
from .target_generator import TargetGenerator, crop_points
from .input_preprocessor import InputPreprocessor


def to_tensor_recursive(obj):
    if isinstance(obj, dict):
        return {k: to_tensor_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_tensor_recursive(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, (int, float)):
        return torch.tensor([obj])
    else:
        return obj
    
def split_by_run_name(df: pd.DataFrame, n_folds=4, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Splits a DataFrame into training and validation sets based on unique experiment_ids.
    Args:
        df: The input DataFrame, expected to have a 'experiment_id' column.
        val_split_ratio: The proportion of experiment_ids to allocate to the validation set.
        random_state: Optional random seed for shuffling experiment_ids for reproducible splits.
    Returns:
        A tuple containing two numpy arrays: train_idx and val_idx.
    """
    if 'experiment_id' not in df.columns:
        raise ValueError("DataFrame must contain a 'experiment_id' column.")
    unique_experiment_ids = df['experiment_id'].unique()
    # Use a local random number generator for shuffling
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_experiment_ids)
    folds = []*n_folds
    for i, experiment_id in enumerate(unique_experiment_ids):
        fold_index = i % n_folds
        folds.append((fold_index, experiment_id))
    return np.array(folds)

def generate_sliding_window_index(copick_root, voxel_spacing, image_size, image_stride, include_edge_windows=True):
    """Generate a sliding window index DataFrame for tomogram crops.
    
    Args:
        copick_root: Initialized CopickRoot object
        voxel_spacing: Voxel spacing to use
        image_size: Size of the crop windows (D, H, W)
        image_stride: Stride for sliding window (D, H, W)
        include_edge_windows: Whether to include edge windows that may go out of bounds
        
    Returns:
        Tuple of (DataFrame, index_array) containing crop information
    """
    df_rows = []
    for run in copick_root.runs:
        voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
        if not voxel_spacing_obj or not voxel_spacing_obj.tomograms:
            print(f"No tomograms found for run {run.name} at voxel spacing {voxel_spacing}")
            continue

        tomogram = voxel_spacing_obj.tomograms[0]
        zarr_path = tomogram.zarr()
        try:
            zarr_array = zarr.open(zarr_path, 'r')['0']
            tomo_shape = zarr_array.shape  # (D, H, W)
        except Exception as e:
            print(f"Error opening zarr array {zarr_path} for run {run.name}: {e}")
            continue

        depth, height, width = tomo_shape
        stride_d, stride_h, stride_w = image_stride
        size_d, size_h, size_w = image_size

        assert depth >= size_d and height >= size_h and width >= size_w, \
            f"Tomogram shape {tomo_shape} is smaller than crop size {image_size}. " \
            f"Please adjust the crop size or use a different tomogram for run {run.name}."

        if include_edge_windows:
            # Calculate the number of crops in each dimension including edge windows
            num_d = ceil((depth - size_d) / stride_d) + 1
            num_h = ceil((height - size_h) / stride_h) + 1
            num_w = ceil((width - size_w) / stride_w) + 1
        else:
            # Only include windows that fit completely within the volume
            num_d = (depth - size_d) // stride_d + 1
            num_h = (height - size_h) // stride_h + 1
            num_w = (width - size_w) // stride_w + 1

        for i in range(num_d):
            for j in range(num_h):
                for k in range(num_w):
                    origin_d = i * stride_d
                    origin_h = j * stride_h
                    origin_w = k * stride_w

                    if include_edge_windows:
                        # Ensure the crop doesn't go out of bounds
                        if origin_d + size_d > depth:
                            origin_d = depth - size_d
                        if origin_h + size_h > height:
                            origin_h = height - size_h
                        if origin_w + size_w > width:
                            origin_w = width - size_w
                    else:
                        # Skip windows that would go out of bounds
                        if (origin_d + size_d > depth or
                            origin_h + size_h > height or
                            origin_w + size_w > width):
                            continue

                    df_rows.append({
                        'experiment_id': run.name,
                        'voxel_spacing': voxel_spacing,
                        'crop_origin_d0': origin_d,
                        'crop_origin_d1': origin_h,
                        'crop_origin_d2': origin_w,
                    })

    df = pd.DataFrame(df_rows)
    if df.empty:
        print("Warning: generate_sliding_window_index generated an empty DataFrame. No valid crops found with the given parameters.")
        return df
    return df

class CropDataset(Dataset):
    """
    This dataset class is designed to be used with the torch DataLoader class.
    This class is supposed to only load the data and feeding it out to the DataLoader.
    Most of the preprocessing should be done in the various Preprocessor/Generator classes, and
    all of the disk operations should be done in the dataset class.
    """

    def __init__(
            self,
            copick_root, # Must be an initialized CopickRoot object
            df,          # DataFrame from DatasetIndexer
            index,       # Index from DatasetIndexer
            voxel_spacing: float,
            image_size: Union[Tuple[int, int, int], int],
            return_targets=False,
            do_augmentation: bool = False, augmentation_args: dict = {}, aurgementation_probabilities: dict = {},
            # input preprocessor
            rescale_factor: float = 1e5, clip_percentile=(0.1, 99.9), standardize=False,
            # target generator
            kernel_size_multiplier: int = 7, kernel_sigma_multiplier: float = 0.5,
            selected_classes: Optional[List[str]] = None
        ):
        self.copick_root = copick_root
        self.df = df
        self.index = index
        self.voxel_spacing = voxel_spacing
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size,) * 3
        self.return_targets = return_targets

        # Initialize caches and copick-derived attributes
        self.points_cache = {}
        # Use selected classes if provided, otherwise use all available classes from copick
        available_classes = [p.name for p in self.copick_root.pickable_objects if p.is_particle]
        if selected_classes is not None:
            invalid_classes = [cls for cls in selected_classes if cls not in available_classes]
            if invalid_classes:
                raise ValueError(f"Selected classes {invalid_classes} not found in copick config. Available: {available_classes}")
            self.classes = selected_classes
        else:
            self.classes = available_classes
        self.particle_radius = {p.name: p.radius for p in self.copick_root.pickable_objects if p.is_particle}
        # print(self.particle_radius, self.classes) # Kept for debugging if needed

        self.input_preprocessor = InputPreprocessor(
            image_size=image_size,
            rescale_factor=rescale_factor,
            clip_percentile=clip_percentile,
            standardize=standardize
        )

        self.aug = do_augmentation
        self.a = augmentation_args
        self.ap = aurgementation_probabilities
        self.augmentator = Augmentator(augmentation_args, aurgementation_probabilities)

        self.target_generator = TargetGenerator(
            target_size=self.image_size,
            classes=self.classes,
            particle_radius=self.particle_radius,
            kernel_size_multiplier=kernel_size_multiplier,
            kernel_sigma_multiplier=kernel_sigma_multiplier,
            voxel_spacing=voxel_spacing
        )

        self.time = False

    # generates a df with the origin of the sliding window crops # Method removed
    # def _generate_df(self, include_edge_windows=True): # Method removed

    def generate_affine_transform(self, crop: np.ndarray, r1: float = 0,
                                  scales: Tuple[float, float, float] = (1, 1, 1), r2: float = 0):

        mat, inv_mat, req_size = generate_affine_mats(np.array(self.image_size), r1, scales, r2)

        # if req_size is larger than the image size and it goes out of bounds, we need to crop the image
        crop_origin, crop_end = crop[:, 0], crop[:, 1]
        crop_origin += ((crop_end - crop_origin) - req_size) // 2
        crop_end = crop_origin + req_size
        crop = np.stack([crop_origin, crop_end], axis=1)
        return mat, inv_mat, crop

    def get_data(self, idx):
        if self.time:
            start = time.time()

        # minimal data preparation
        df_idx = self.index[idx]
        entry = self.df.loc[df_idx]
        applied_augmentations = {}
        run_name = entry['experiment_id']
        tomo_key = (run_name, self.voxel_spacing)

        # Load tomogram from copick
        run = self.copick_root.get_run(run_name)
        voxel_spacing_obj = run.get_voxel_spacing(entry['voxel_spacing'])
        tomogram = [t for t in voxel_spacing_obj.tomograms if t.static_path.endswith('denoised.zarr')][0]
        # arr is not fully loaded into ram yet
        image = zarr.open(tomogram.zarr(), 'r')['0']

        if self.time:
            print(f"Loaded tomogram for {tomo_key}", time.time() - start)

        crop_origin = np.array([entry['crop_origin_d0'], entry['crop_origin_d1'], entry['crop_origin_d2']])
        if self.aug and 'shift' in self.ap and np.random.rand() < self.ap['shift']:
            image_shape = image.shape
            shift_max = np.minimum(image_shape - crop_origin - self.image_size,
                                   [self.a['shift'][f'd{i}'][1] for i in range(3)]) + 1
            shift_min = np.maximum(0, [self.a['shift'][f'd{i}'][0] for i in range(3)])
            shift_amount = np.random.randint(shift_min, shift_max)
            applied_augmentations['shift'] = shift_amount
            crop_origin = crop_origin + shift_amount
        crop_end = crop_origin + np.asarray(self.image_size)
        crop = np.stack([crop_origin, crop_end], axis=1)  # (3, 2)

        if self.aug and 'affine' in self.ap and np.random.rand() < self.ap['affine']:
            if 'rotate' not in self.a['affine']:
                r1, r2 = np.random.uniform(0, 2 * np.pi, 2)
            else:
                r1, r2 = np.random.uniform(self.a['affine']['rotate'][0], self.a['affine']['rotate'][1], 2)
            scales = np.random.uniform(self.a['affine']['low'], self.a['affine']['high'], 3) * np.random.choice([-1, 1], 3)
            applied_augmentations['affine'] = (r1, scales, r2)
            mat, inv_mat, crop = self.generate_affine_transform(crop, r1, scales, r2)

        if self.time:
            print('crop, affine done', applied_augmentations, time.time() - start)

        image = self.input_preprocessor(image, crop=crop, inv_affine=inv_mat if 'affine' in applied_augmentations else None)

        if self.time:
            print('preprocessing done', time.time() - start)

        if not self.return_targets:
            if self.aug:
                image = self.augmentator(image=image, target=None, applied_augmentations=applied_augmentations)['image']
            return torch.from_numpy(image[None]), None

        # Load points from cache or disk
        points = {}
        run = self.copick_root.get_run(run_name)
        for pick in run.get_picks():
            pick_points, _ = pick.numpy()
            if len(pick_points) > 0:
                points_in_voxels = pick_points[:, ::-1] / self.voxel_spacing
                points[pick.pickable_object_name] = points_in_voxels
            else:
                points[pick.pickable_object_name] = np.empty((0, 3), dtype=np.float32)

        if self.time:
            print('points loaded', time.time() - start)

        target = self.target_generator(points, crop, affine=mat if 'affine' in applied_augmentations else None)

        if self.aug:
            out = self.augmentator(image=image, target=target, applied_augmentations=applied_augmentations)
            image, target = out['image'], out['target']

        if self.time:
            print('augmentation done', time.time() - start)
            print('applied augmentations:', applied_augmentations)
        target = to_tensor_recursive(target)
        return torch.from_numpy(image[None]), target

    def __getitem__(self, idx):
        # any kind of multi-image augmentation should be done here
        if self.aug and 'fuse' in self.ap:
            th = np.random.rand()
            if th < self.ap['fuse']['mixup']:
                if not self.return_targets:
                    raise ValueError('Mixup should not be used without targets')
                idx2 = np.random.randint(len(self.index))
                image1, target1 = self.get_data(idx)
                image2, target2 = self.get_data(idx2)
                if 'mixup' in self.a:
                    beta_alpha = self.a['mixup']['alpha']
                else:
                    beta_alpha = 1
                lam = float(np.random.beta(beta_alpha, beta_alpha))
                lam = max(lam, 1 - lam)  # lam should be between 0.5 and 1 to focus on the first image
                image = lam * image1 + (1 - lam) * image2

                # target = {k: lam * v1 + (1 - lam) * v2 for k, v1, v2 in zip(target1.keys(), target1.values(), target2.values())}
                target1['heatmap'] = lam * target1['heatmap'] + (1 - lam) * target2['heatmap']
                return image, target1
            elif th < self.ap['fuse']['cutmix'] + self.ap['fuse']['mixup']:
                if not self.return_targets:
                    raise ValueError('Cutmix should not be used without targets')
                idx2 = np.random.randint(len(self.index))
                image1, target1 = self.get_data(idx)
                image2, target2 = self.get_data(idx2)

                l0, l1, l2 = np.random.randint(self.a['cutmix']['min'], self.a['cutmix']['max'], size=3)
                s0 = np.random.randint(0, self.image_size[0] - l0)
                s1 = np.random.randint(0, self.image_size[1] - l1)
                s2 = np.random.randint(0, self.image_size[2] - l2)
                image1[:, s0:s0 + l0, s1:s1 + l1, s2:s2 + l2] = image2[:, s0:s0 + l0, s1:s1 + l1, s2:s2 + l2]

                target1['heatmap'][:, s0:s0 + l0, s1:s1 + l1, s2:s2 + l2] = target2['heatmap'][:, s0:s0 + l0, s1:s1 + l1, s2:s2 + l2]
                return image1, target1
            else:
                return self.get_data(idx)
        else:
            return self.get_data(idx)

    def __len__(self):
        return len(self.index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def batch_points(self, points):
        out = {}
        for cls in self.classes:
            to_cat = []
            for b, p in enumerate(points):
                if cls not in p:
                    continue
                to_cat.append(torch.cat([torch.full((p[cls].shape[0], 1), b), p[cls]], dim=1))
            if len(to_cat) > 0:
                out[cls] = torch.cat(to_cat, dim=0)
        return out

    def collate_fn(self, batch):
        out = tuple(zip(*batch))
        images, targets = out
        images = torch.stack(images)

        if targets[0] is None:
            return images, None

        batched_targets = {}
        for k in targets[0].keys():
            if k == 'points':
                batched_targets[k] = self.batch_points([t[k] for t in targets])
            else:
                batched_targets[k] = torch.stack(
                    [t[k] for t in targets]
                )
        return images, batched_targets

def build_dataloaders(
    copick_root,
    df,
    train_index,
    val_index,
    cfg,
    val_df=None
):
    train_ds = CropDataset(
        copick_root = copick_root,
        df = df,
        index = train_index,
        voxel_spacing = cfg['dataset']['voxel_spacing'],
        image_size = cfg['dataset']['image_size'],
        return_targets = True,
        do_augmentation = True,
        augmentation_args = cfg['augmentation_args'],
        aurgementation_probabilities = cfg['augmentation_probabilities'],
        selected_classes = cfg['dataset']['classes'],
    )
    val_ds = CropDataset(
        copick_root = copick_root,
        df = df if val_df is None else val_df,
        index = val_index,
        voxel_spacing = cfg['dataset']['voxel_spacing'],
        image_size = cfg['dataset']['image_size'],
        return_targets = True,
        do_augmentation = False,  # No augmentation for validation
        selected_classes = cfg['dataset']['classes'],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg['train_loop']['train_batch_size'],
        shuffle=True,
        num_workers=cfg['system']['num_workers'],
        collate_fn=train_ds.collate_fn,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg['train_loop']['val_batch_size'],
        shuffle=False,
        num_workers=cfg['system']['num_workers'],
        collate_fn=val_ds.collate_fn,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader
