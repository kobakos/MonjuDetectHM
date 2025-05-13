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

#from models.detr.util.misc import NestedTensor
from .augmentations import Augmentator, generate_affine_mats
from .target_generator import TargetGenerator, crop_points
from .input_preprocessor import InputPreprocessor

def get_image(path, resolution_hierarchy=0):
    assert os.path.exists(path), path
    if "processed" in path:
        resolution_hierarchy = None
    image = zarr.open(path)
    image = image[resolution_hierarchy] if resolution_hierarchy is not None else image
    return image

def get_points(experiment_id, json_path):
    with open(json_path) as f:
        annotation_dict = json.load(f)
    points = []
    points_dicts = annotation_dict['points']
    for d in points_dicts:
        points.append([d['location']['z'], d['location']['y'], d['location']['x']])
    return np.array(points)

def get_points_ext(experiment_id, jsonl_path):
    with open(
            jsonl_path
        ) as f:
        annotation_dicts = ndjson.load(f)
    points = []
    for d in annotation_dicts:
        points.append([d['location']['z'], d['location']['y'], d['location']['x']])
    return np.array(points)*10.012444 # multiply by 10 to convert to angstrom

def make_mix_func_linear(end_wbp, end_ctf, wbf_mul = 1, ctf_mul = 1):
    """
    returns a function, that takes epoch as input and returns a dict:
    {
        "denoised" : alpha,
        "ctfdeconvolved" : beta,
        "wbp": gamma
    }
    """
    def f(epoch):
        if epoch <= end_wbp:
            return {
                "denoised": 0,
                "ctfdeconvolved": (epoch) / end_wbp * ctf_mul,
                "wbp": (end_wbp - epoch) / end_wbp * wbf_mul,
            }
        elif epoch <= end_ctf:
            return {
                "denoised": (epoch - end_wbp) / (end_ctf - end_wbp),
                "ctfdeconvolved": (end_ctf - epoch) / (end_ctf - end_wbp) * ctf_mul,
                "wbp": 0
            }
        else:
            return {
                "denoised": 1,
                "ctfdeconvolved": 0,
                "wbp": 0
            }
    return f

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
    
def combine_points(point_dict1, point_dict2):
    out = {}
    for cls in set(point_dict1.keys()).union(set(point_dict2.keys())):
        if cls not in point_dict1:
            out[cls] = point_dict2[cls]
        elif cls not in point_dict2:
            out[cls] = point_dict1[cls]
        else:
            if torch.is_tensor(point_dict1[cls]):
                out[cls] = torch.cat([point_dict1[cls], point_dict2[cls]], dim=0)
            else:
                out[cls] = np.concatenate([point_dict1[cls], point_dict2[cls]], axis=0)
    return out

def combine_points_cutmix(point_dict1, point_dict2, start, end, margin=5):
    """
    point_dict1: dict of points from the first image (base image)
    point_dict2: dict of points from the second image (image to be cutmixed)
    """
    points_in = {}# points inside the cutmix region
    points_out = {}# points outside the cutmix region
    for cls in set(point_dict1.keys()).union(set(point_dict2.keys())):
        if cls not in point_dict1:
            mask = ((point_dict2[cls] > (start[None] - margin)) & (point_dict2[cls] < (end[None] + margin))).all(axis=-1)
            if mask.any():
                points_in[cls] = point_dict2[cls][mask]
        elif cls not in point_dict2:
            mask = ((point_dict1[cls] < (start[None] + margin)) | (point_dict1[cls] > (end[None] - margin))).any(axis=-1)
            if mask.any():
                points_out[cls] = point_dict1[cls][mask]
        else:
            mask1 = ((point_dict1[cls] < (start[None] + margin)) | (point_dict1[cls] > (end[None] - margin))).any(axis=-1)
            mask2 = ((point_dict2[cls] > (start[None] - margin)) & (point_dict2[cls] < (end[None] + margin))).all(axis=-1)
            if mask1.any():
                points_out[cls] = point_dict1[cls][mask1]
            if mask2.any():
                points_in[cls] = point_dict2[cls][mask2]
    return combine_points(points_out, points_in)
        

class dataset(Dataset):# should support test data as well
    """
    This dataset class is designed to be used with the torch DataLoader class.
    This class is supposed to only load the data and feeding it out to the DataLoader.
    Most of the preprocessing should be done in the various Preprocessor/Generator classes, and
    all of the disk operations should be done in the dataset class.
    """
    def __init__(self,
            index,df,image_path: Union[Path, str],return_targets=False,
            do_augmentation: bool = False, augmentation_args: dict = {}, aurgementation_probabilities: dict = {}, path_modifier = None,
            # dataset specific arguments
            external_path: Union[Path, str]=None,
            image_size: Tuple[int, int, int] = (128, 128, 128), resolution_hierarchy: int = 0,
            # input preprocessor
            rescale_factor: float = 1e5, clip_percentile = (0.1, 99.9), standardize = False,
            # target generator
            kernel_size_multiplier: int = 7, kernel_sigma_multiplier: float = 0.5,
            detr: bool = False, return_num_points: bool = False,
            generate_offset_target: bool = False, offset_radius_multiplier: float = 0.1,

            domain_blend_params=None, epoch=None, in_memory=False

        ):
        self.index = index
        self.df = df
        self.image_base_path = Path(image_path)
        self.external_path = Path(external_path) if external_path is not None else None
        self.return_targets = return_targets
        self.detr = detr
        self.path_modifier = path_modifier

        self.input_preprocessor = InputPreprocessor(
            image_size = image_size,
            rescale_factor = rescale_factor,
            clip_percentile = clip_percentile,
            standardize = standardize
        )
        
        self.aug = do_augmentation
        self.a = augmentation_args
        self.ap = aurgementation_probabilities
        self.augmentator = Augmentator(augmentation_args, aurgementation_probabilities)
        
    
        self.image_size = image_size
        self.original_image_sizes = {
            'original': (184, 630, 630),
            'external': (200, 630, 630)
        }
        self.resolution_hierarchy = resolution_hierarchy
        self.pixel_spacing = [
            10.012444196428572, 
            10.012444196428572*2,
            10.012444196428572*4][resolution_hierarchy]

        self.classes = [
            'apo-ferritin',
            # 'beta-amylase', # ignore beta-amylase for now as it is deemed impossible to detect
            'beta-galactosidase',
            'ribosome',
            'thyroglobulin',
            'virus-like-particle'
        ]
        particle_radius = {
            'apo-ferritin': 60,  # apo-ferritin
            'beta-amylase': 65,  # beta-amylase
            'beta-galactosidase': 90,  # beta-galactosidase
            'ribosome': 150,  # ribosome
            'thyroglobulin': 130,  # thyroglobulin
            'virus-like-particle': 135,  # virus-like-particle
        }
        particle_radius = {k: round(v / self.pixel_spacing) for k, v in particle_radius.items()}

        weights = {
            'apo-ferritin': 1,
            'beta-amylase': 0,
            'beta-galactosidase': 2,
            'ribosome': 1,
            'thyroglobulin': 2,
            'virus-like-particle': 1,
        }

        self.cls2ext_path = {
            'apo-ferritin': '101/ferritin_complex-1.0_orientedpoint.ndjson',
            'beta-amylase': '102/beta_amylase-1.0_orientedpoint.ndjson',
            'beta-galactosidase': '103/beta_galactosidase-1.0_orientedpoint.ndjson',
            'ribosome': '104/cytosolic_ribosome-1.0_orientedpoint.ndjson',
            'thyroglobulin': '105/thyroglobulin-1.0_orientedpoint.ndjson',
            'virus-like-particle': '106/pp7_vlp-1.0_orientedpoint.ndjson',
        }

        self.target_generator = TargetGenerator(
            target_size = image_size,
            classes = self.classes,
            particle_radius = particle_radius,
            kernel_size_multiplier=kernel_size_multiplier,
            kernel_sigma_multiplier=kernel_sigma_multiplier,
            return_as_points=self.detr,
            return_num_points=return_num_points,
            generate_offset_target=generate_offset_target,
            offset_radius_multiplier=offset_radius_multiplier
        )
        self.return_num_points = return_num_points

        self.domain_blend = domain_blend_params is not None 
        if self.domain_blend:
            self.mix_func = make_mix_func_linear(
                domain_blend_params['end_wbp'],
                domain_blend_params['end_ctf']
            )

        self.epoch = epoch

        self.in_memory = in_memory
        self.images = {}

        self.time = False

    def generate_affine_transform(self, crop: np.ndarray, r1: float = 0, scales: Tuple[float, float, float] = (1, 1, 1), r2: float = 0):

        mat, inv_mat, req_size = generate_affine_mats(np.array(self.image_size), r1, scales, r2)

        # if req_size is larger than the image size and it goes out of bounds, we need to crop the image
        crop_origin, crop_end = crop[:, 0], crop[:, 1]
        crop_origin += ((crop_end - crop_origin) - req_size) // 2 
        crop_end = crop_origin + req_size
        crop = np.stack([crop_origin, crop_end], axis=1)
        return mat, inv_mat, crop
    
    def get_image(self, path, resolution_hierarchy):
        if self.in_memory and path in self.images:
            return self.images[path]
        else:
            image = get_image(path, resolution_hierarchy)
            if self.in_memory:
                self.images[path] = np.asarray(image)
            return image

    def get_data(self, idx):
        if self.time:
            start = time.time()
        #print('loading image', idx)

        # minimal data preparation
        df_idx = self.index[idx]
        entry = self.df.loc[df_idx]
        applied_augmentations = {}
        
        zarr_path = entry['zarr_path']
        if self.path_modifier is not None:
            zarr_path = self.path_modifier(zarr_path)
        
        if self.aug and 'random_zarr' in self.ap and np.random.rand() < self.ap['random_zarr']:
            assert entry['source'] == 'original', 'attempting to apply random_zarr to external data'
            to = random.choice(self.a['random_zarr']['possible_postprocessings'])
            zarr_path = zarr_path.replace('denoized', to)
            applied_augmentations['random_zarr'] = to
        
        if self.domain_blend:
            blend = self.mix_func(self.epoch)
            image = {}
            for n in ['wbp', 'ctfdeconvolved', 'denoised']:
                if blend[n] > 0:
                    image[n] = {
                        'weight': blend[n],
                        'image': self.get_image(zarr_path.replace('denoized', n), self.resolution_hierarchy)
                    }
        else:
            image = self.get_image(zarr_path, self.resolution_hierarchy)
        if self.time:
            print('loaded image', time.time()-start)

        crop_origin =  np.array([entry['crop_origin_d0'], entry['crop_origin_d1'], entry['crop_origin_d2']])
        if self.aug and 'shift' in self.ap and np.random.rand() < self.ap['shift']:
            if isinstance(image, dict):
                image_shape = (184 ,630, 630)
            else:
                image_shape = image.shape
            shift_max = np.minimum(image_shape - crop_origin - self.image_size, [self.a['shift'][f'd{i}'][1] for i in range(3)]) + 1
            shift_min = np.maximum(0, [self.a['shift'][f'd{i}'][0] for i in range(3)])
            shift_amount = np.random.randint(shift_min, shift_max)
            applied_augmentations['shift'] = shift_amount
            crop_origin = crop_origin + shift_amount
        crop_end = crop_origin + np.asarray(self.image_size)
        crop = np.stack([crop_origin, crop_end], axis=1)# (3, 2)

        if self.aug and 'affine' in self.ap and np.random.rand() < self.ap['affine']:
            if 'rotate' not in self.a['affine']:
                r1, r2 = np.random.uniform(0, 2*np.pi, 2)
            else:
                r1, r2 = np.random.uniform(self.a['affine']['rotate'][0], self.a['affine']['rotate'][1], 2)   
            scales = np.random.uniform(self.a['affine']['low'], self.a['affine']['high'], 3) * np.random.choice([-1, 1], 3)
            applied_augmentations['affine'] = (r1, scales, r2)
            mat, inv_mat, crop = self.generate_affine_transform(crop, r1, scales, r2)

        if self.time:
            print('crop, affine done', applied_augmentations, time.time()-start)

        image = self.input_preprocessor(image, crop=crop, inv_affine=inv_mat if 'affine' in applied_augmentations else None)

        if self.time:
            print('preprocessing done', time.time()-start)

        if not self.return_targets:
            if self.aug:
                image = self.augmentator(image = image, target = None, applied_augmentations=applied_augmentations)['image']
            return torch.from_numpy(image[None]), None
        
        points = {}
        for cls in self.classes:
            if entry['source'] == 'external':
                points[cls] = get_points_ext(entry['experiment_id'], self.external_path/'Simulated'/entry['experiment_id']/'Reconstructions'/'VoxelSpacing10.000'/'Annotations'/self.cls2ext_path[cls])
            elif entry['source'] == 'original':
                points[cls] = get_points(entry['experiment_id'], self.image_base_path/'overlay'/'ExperimentRuns'/entry['experiment_id']/'Picks'/f'{cls}.json')

        if self.time:
            print('points loaded', time.time()-start)

        target = self.target_generator(points, crop, affine = mat if 'affine' in applied_augmentations else None)
        target["domain"] = float(entry['source'] == 'external')

        if self.aug:
            out = self.augmentator(image = image, target = target, applied_augmentations=applied_augmentations)
            image, target = out['image'], out['target']

        if self.time:
            print('augmentation done', time.time()-start)
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
                lam = max(lam, 1 - lam)# lam should be between 0.5 and 1 to focus on the first image
                image = lam * image1 + (1 - lam) * image2
                
                #target = {k: lam * v1 + (1 - lam) * v2 for k, v1, v2 in zip(target1.keys(), target1.values(), target2.values())}
                target1['heatmap'] = lam * target1['heatmap'] + (1 - lam) * target2['heatmap']
                if 'num_points' in target1:
                    target1['num_points'] = lam * target1['num_points'] + (1 - lam) * target2['num_points']
                if 'domain' in target1:
                    target1['domain'] = lam * target1['domain'] + (1 - lam) * target2['domain']
                if 'offset' in target1:
                    target1['offset'] = torch.minimum(target1['offset'], target2['offset'])
                if 'points' in target1:
                    target1['points'] = combine_points(target1['points'], target2['points'])
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
                image1[:, s0:s0+l0, s1:s1+l1, s2:s2+l2] = image2[:, s0:s0+l0, s1:s1+l1, s2:s2+l2]
                
                target1['heatmap'][:, s0:s0+l0, s1:s1+l1, s2:s2+l2] = target2['heatmap'][:, s0:s0+l0, s1:s1+l1, s2:s2+l2]
                lam = (1 - (l0 * l1 * l2 / image1.numel())).item()
                if 'num_points' in target1:
                    target1['num_points'] = lam * target1['num_points'] + (1 - lam) * target2['num_points']
                if 'domain' in target1:
                    target1['domain'] = lam * target1['domain'] + (1 - lam) * target2['domain']
                if 'offset' in target1:
                    target1['offset'][:, :, s0:s0+l0, s1:s1+l1, s2:s2+l2] = target2['offset'][:, :, s0:s0+l0, s1:s1+l1, s2:s2+l2]
                if 'points' in target1:
                    target1['points'] = combine_points_cutmix(target1['points'], target2['points'], torch.tensor([s0, s1, s2]), torch.tensor([s0+l0, s1+l1, s2+l2]))
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


    #@staticmethod
    #def worker_init_fn(worker_id):
    #    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    @staticmethod
    def create_worker_init_fn(epoch):
        seed = np.random.get_state()[1][0] + epoch*32
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)
        return worker_init_fn

# Faster when shuffle==False
class SlidingWindowDataset(Dataset):
    def __init__(self,
            index,df,image_path: Union[Path, str] = Path('../data/original'),return_targets=False,
            do_augmentation: bool = False, augmentation_args: dict = {}, aurgementation_probabilities: dict = {}, path_modifier = None,
            # dataset specific arguments
            external_path: Union[Path, str] = Path('../data/external'),
            image_size: Tuple[int, int, int] = (128, 128, 128), resolution_hierarchy: int = 0,
            # input preprocessor
            rescale_factor: float = 1e5, clip_percentile = (0.1, 99.9), standardize = False,
            # target generator
            kernel_size_multiplier: int = 7, kernel_sigma_multiplier: float = 0.5,
            detr: bool = False, return_num_points: bool = False,
            epoch=None, domain_blend_params=None, del_cached=True,
            **kwargs
        ):
        if detr or return_num_points:
            raise ValueError('SlidingWindowDataset does not support DETR or return_num_points')
        image_sizes = {
            0: (184, 630, 630),
            1: (92, 315, 315),
            2: (46, 157, 157),
        }
        #self.inner_df = df[
        #    (df['crop_origin_d0'] == 0) & (df['crop_origin_d1'] == 0) & (df['crop_origin_d2'] == 0)
        #]
        self.inner_df = df.drop_duplicates(subset=['experiment_id'])
        self.inner_df.loc[:, 'crop_origin_d0'] = 0
        self.inner_df.loc[:, 'crop_origin_d1'] = 0
        self.inner_df.loc[:, 'crop_origin_d2'] = 0
        #self.inner_df = df.loc[df.groupby('experiment_id')['crop_origin_d0'].idxmin()]
        #self.inner_df = df.loc[df.groupby('experiment_id')[['crop_origin_d0', 'crop_origin_d1', 'crop_origin_d2']].idxmin().min(axis=1)]
        #self.inner_df.set_index('experiment_id', inplace=True, drop=False)
        #print(self.inner_df)
        self.inner_df_index = self.inner_df.index.values
        self.experiment_ids = list(self.inner_df['experiment_id'])
        self.ds = dataset(
            self.inner_df_index, self.inner_df, image_path, return_targets, False, {}, {}, path_modifier,
            external_path, image_sizes[resolution_hierarchy], resolution_hierarchy,
            rescale_factor, clip_percentile, standardize,
            kernel_size_multiplier, kernel_sigma_multiplier, False, False,
            epoch=epoch, domain_blend_params=domain_blend_params, **kwargs
        )
        self.image_size = np.asarray(image_size)
        self.index = index

        self.cached_images = {}
        self.cached_targets = {}
        self.num_access = {}
        self.df = df
        self.tiles_per_experiment = len(self.df[self.df['experiment_id'] == self.df['experiment_id'].iloc[0]])
        self.del_cached = del_cached
    
    def __getitem__(self, idx):
        idx = self.index[idx]
        entry = self.df.loc[idx]
        experiment_id = entry['experiment_id'] 
        if experiment_id not in self.cached_images:
            #print(self.ds[idx])
            image, target = self.ds[self.experiment_ids.index(experiment_id)]
            self.cached_images[experiment_id] = image
            if self.image_size[0] > 184:
                self.cached_images[experiment_id] = torch.nn.functional.pad(self.cached_images[experiment_id], (
                    0, 0,
                    0, 0,
                    0, self.image_size[0] - 184,
                ))
            self.cached_targets[experiment_id] = target
            self.num_access[experiment_id] = 0

        crop_start = entry['crop_origin_d0'], entry['crop_origin_d1'], entry['crop_origin_d2']
        image = self.cached_images[experiment_id][
            :,
            crop_start[0]:crop_start[0]+self.image_size[0],
            crop_start[1]:crop_start[1]+self.image_size[1],
            crop_start[2]:crop_start[2]+self.image_size[2]
        ]
        target = self.cached_targets[experiment_id]['heatmap'][
            :,
            crop_start[0]:crop_start[0]+self.image_size[0],
            crop_start[1]:crop_start[1]+self.image_size[1],
            crop_start[2]:crop_start[2]+self.image_size[2]
        ]
        target = {'heatmap': target, 'domain': self.cached_targets[experiment_id]['domain']}
        self.num_access[experiment_id] += 1
        if self.num_access[experiment_id] >= self.tiles_per_experiment and self.del_cached:
            del self.cached_images[experiment_id]
            del self.cached_targets[experiment_id]
            del self.num_access[experiment_id]
        return image, target
    def __len__(self):
        return len(self.index)
    @staticmethod
    def collate_fn(batch):
        out = tuple(zip(*batch))
        images, targets = out
        images = torch.stack(images)
        if targets[0] is None:
            return images, None
        targets = {k: torch.stack([t[k] for t in targets]) for k in targets[0].keys()}
        return images, targets
    
class SlidingWindowDataloader():
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.index = np.arange(len(self.dataset))
        self.drop_last = False
        self.index = np.concatenate([self.index] * self.prefetch_factor)
        self.prefetch_index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.prefetch_index >= len(self.index):
            self.prefetch_index = 0
            raise StopIteration()
        batch = self.index[self.prefetch_index:self.prefetch_index+self.batch_size]
        self.prefetch_index += self.batch_size
        return self.dataset.collate_fn([self.dataset[i] for i in batch])
    def __len__(self):
        return len(self.index) // self.batch_size + int(len(self.index) % self.batch_size != 0)
     

import matplotlib.pyplot as plt
class dataset_visualizer(dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = {
            'apo-ferritin': np.array((255, 0, 0)),
            'beta-amylase': np.array((0, 255, 0)),
            'beta-galactosidase': np.array((0, 0, 255)),
            'ribosome': np.array((255, 255, 0)),
            'thyroglobulin': np.array((255, 0, 255)),
            'virus-like-particle': np.array((0, 255, 255)),
        }
        os.makedirs('visualization_results', exist_ok=True)
    def show_histogram(self, idx):
        os.makedirs(f'visualization_results/{idx}', exist_ok=True)
        image, _ = self[idx]
        image = image.squeeze().numpy()
        plt.hist(image.flatten(), bins=100)
        plt.savefig(f'visualization_results/{idx}/histogram.png')
    def visualize(self, idx, axis=0, step_size=25, save_dir: str = None):
        if save_dir is None:
            save_dir = str(idx)
        os.makedirs(f'visualization_results/{save_dir}', exist_ok=True)
        image, target = self[idx]
        image = image.squeeze().numpy()
        if target is not None:
            target = target.squeeze().numpy()
        for i in range(0, image.shape[axis], step_size):
            if axis == 0:
                im = image[i]
                if target is not None:
                    tar = target[:, i]
            elif axis == 1:
                im = image[:, i]
                if target is not None:
                    tar = target[:, :, i]
            elif axis == 2:
                im = image[:, :, i]
                if target is not None:
                    tar = target[:, :, :, i]
            #im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            im = (im + 2) / 4 * 255
            im = im[:, :, None].repeat(3, axis=2).astype(np.uint8)
            if target is not None:
                tar_cls = np.zeros_like(im)
                for cls_idx in range(len(self.classes)):
                    cls = self.classes[cls_idx]
                    #tar_cls = tar[cls_idx]
                    tar_cls = np.maximum(tar_cls, tar[cls_idx, :, :][:, :, None]*self.colors[cls][None, None, :])
                    #tar_cls = (tar_cls + 1) / 2 # normalize to 0.5 - 1
                tar_cls = np.concatenate([im, tar_cls], axis=1)
                cv2.imwrite(f'visualization_results/{save_dir}/{i}.png', tar_cls)
            else:
                cv2.imwrite(f'visualization_results/{save_dir}/{i}.png', im)
        return image

def build_dataloaders(train_df, val_df, fold, cfg, settings, train_index=None, pretrain=False, epoch=0):
    B = Path(settings['base_path'])
    if train_index is not None:
        pass
    elif pretrain:
        train_index = train_df.index[(train_df['fold'] != fold) * (train_df['source'] == 'external')].values
    else:
        train_index = train_df.index[(train_df['fold'] != fold) * (train_df['source'] != 'external')].values

    train_ds = dataset(
        index = train_index,
        df = train_df,
        return_targets=True,
        image_path=B/settings['competition_data_path']/'train',
        external_path=B/settings['external_data_path'],
        do_augmentation=True,
        augmentation_args=cfg['augmentation_args'],
        aurgementation_probabilities=cfg['augmentation_probabilities'],
        **cfg['feature_extractor_args'],
        epoch=epoch
    )
    #val_ds = dataset(
    #    index = val_df.index[val_df['fold'] == fold].values,
    #    df = val_df,
    #    return_targets=True,
    #    image_path = cfg['paths']['data_path'] + '/train',
    #    **cfg['feature_extractor_args']
    #)
    val_cfg = cfg.copy()
    if pretrain:
        val_index = val_df.index[(val_df['fold'] == fold) * val_df['source'] == 'external'].values
    else:
        val_index = val_df.index[(val_df['fold'] == fold) * (val_df['source'] != 'external')].values
    val_ds = SlidingWindowDataset(
        index = val_index,
        df = val_df,
        return_targets=True,
        image_path=B/settings['competition_data_path']/'train',
        external_path=B/settings['external_data_path'],
        **cfg['feature_extractor_args'],
        epoch=epoch
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size = cfg['train_loop']['train_batch_size'],
        shuffle = True,
        num_workers = cfg['system']['num_workers'] if not cfg['feature_extractor_args'].get('in_memory', False) else 0,
        prefetch_factor = cfg['system']['prefetch_factor'],
        collate_fn = train_ds.collate_fn,
        worker_init_fn = train_ds.create_worker_init_fn(epoch),
        pin_memory = True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size = cfg['train_loop']['val_batch_size'],
        shuffle = False,
        num_workers = 0,
        prefetch_factor = None,#cfg['system']['prefetch_factor'],
        collate_fn = val_ds.collate_fn,
        pin_memory = True
    )
    return train_dl, val_dl
