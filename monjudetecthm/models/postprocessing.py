from typing import List, Union, Tuple, Optional
import concurrent.futures
from math import ceil

import torch
import torch.nn.functional as F
import numpy as np
import zarr

from scipy import ndimage
import copick

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_ndimage = None

# copied from hengck23's kernel https://www.kaggle.com/code/hengck23/speed-up-connected-component-analysis-with-pytorch
#faster version using cuda

#https://github.com/kornia/kornia/blob/9ccae8c297a00a35d811b5a6e4f468a1d54d17f4/kornia/contrib/connected_components.py#L7
#https://stackoverflow.com/questions/46840707/efficiently-find-centroid-of-labelled-image-regions

def find_connecte_component(probability, threshold, max_radius = 10):
    device = probability.device
    probability = probability.detach()
    num_particle_type, D, H, W = probability.shape
    mask = probability > torch.tensor(threshold,device=device).reshape(num_particle_type,1,1,1)

    # allocate the output tensors for labels
    out = (torch.arange(D * H * W, device=device, dtype=torch.float32)+1).reshape(1,D, H, W)
    out = out.repeat(num_particle_type,1,1,1)
    out[~mask] = 0

    out = out.reshape(num_particle_type,1,D, H, W)
    mask = mask.reshape(num_particle_type,1,D, H, W)
    for _ in range(max_radius):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication
    out = out.reshape(num_particle_type,D,H,W)
    out = out.long()
    component=[]
    for i in range(num_particle_type):
        u, inverse = torch.unique(out[i], sorted=True, return_inverse=True)
        component.append(inverse)
    component = torch.stack(component)
    #plt.imshow(component[1].data.cpu().numpy().max(0))
    return component
    
def find_centroid(component):
    device = component.device
    num_particle_type, D, H, W = component.shape
    count = component.flatten(1).max(-1)[0]+1
    cumcount = torch.zeros(num_particle_type+1, dtype=torch.int32, device=device)
    cumcount[1:] = torch.cumsum(count,0)
    component = component+cumcount[:-1].reshape(num_particle_type,1,1,1)

    gridz = torch.arange(0, D, device=device).reshape(1,D,1,1).expand(num_particle_type,-1,H,W)
    gridy = torch.arange(0, H, device=device).reshape(1,1,H,1).expand(num_particle_type,D,-1,W)
    gridx = torch.arange(0, W, device=device).reshape(1,1,1,W).expand(num_particle_type,D,H,-1)
    n  = torch.bincount(component.flatten())
    nx = torch.bincount(component.flatten(),weights=gridx.flatten())
    ny = torch.bincount(component.flatten(),weights=gridy.flatten())
    nz = torch.bincount(component.flatten(),weights=gridz.flatten())

    x=nx/n
    y=ny/n
    z=nz/n
    xyz = torch.stack([x,y,z],1).float()
    xyz = torch.split(xyz, count.tolist(), dim=0)
    centroid = [xxyyzz[1:] for xxyyzz in xyz]
    return centroid


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))

def gaussian_kernel(sigma, size):
    """
    indices: np.array with shape (3, n)
    center: np.array with shape (3,)
    sigma: float
    """
    size = np.asarray(size)
    indices = np.indices(size)
    return np.exp(-np.sum((indices - size[:, None, None, None]//2) ** 2, axis=0) / (2 * sigma ** 2))

def create_weight(size, edge_begin = 0.25, min_weight = 0.33333):
    """
    size: 3-length tuple
    edge_begin: float
    minimum: float
    """
    weight = np.ones(size)
    slope0 = np.linspace(min_weight, 1, round(size[0] * edge_begin))
    slope1 = np.linspace(min_weight, 1, round(size[1] * edge_begin))
    slope2 = np.linspace(min_weight, 1, round(size[2] * edge_begin))

    weight[:round(edge_begin * size[0]), :, :] = np.minimum(weight[:round(edge_begin * size[0]), :, :], slope0[:, None, None])
    weight[-round(edge_begin * size[0]):, :, :] = np.minimum(weight[-round(edge_begin * size[0]):, :, :], slope0[::-1, None, None])

    weight[:, :round(edge_begin * size[1]), :] = np.minimum(weight[:, :round(edge_begin * size[1]), :], slope1[None, :, None])
    weight[:, -round(edge_begin * size[1]):, :] = np.minimum(weight[:, -round(edge_begin * size[1]):, :], slope1[None, ::-1, None])

    weight[:, :, :round(edge_begin * size[2])] = np.minimum(weight[:, :, :round(edge_begin * size[2])], slope2[None, None, :])
    weight[:, :, -round(edge_begin * size[2]):] = np.minimum(weight[:, :, -round(edge_begin * size[2]):], slope2[None, None, ::-1])
    return weight

def hmap_to_points_ccl(hmap: np.ndarray, threshold:np.ndarray):
    """
    hmap: np.array with shape (D, H, W)
    threshold: float
    """
    #hmap = cp.asarray(hmap)
    hmap = hmap.float().cpu().numpy()
    #hmap = ndimage.gaussian_filter(hmap, 1)
    binary = hmap > threshold
    label, n_obj = ndimage.label(binary)
    if n_obj == 0:
        return {"points": []}
    points = ndimage.center_of_mass(hmap, label, range(1, n_obj+1))
    points = np.stack(points)
    #breakpoint()
    #points = cp.stack(points)
    #points = points.get()
    return {"points": points}

def hmap_to_points_ccl_gpu(hmap: np.ndarray, threshold:np.ndarray):
    """
    hmap: np.array with shape (D, H, W)
    threshold: float
    """
    hmap = cp.asarray(hmap)
    #hmap = hmap.float().cpu().numpy()
    hmap = cupy_ndimage.gaussian_filter(hmap, 1)
    binary = hmap > threshold
    label, n_obj = cupy_ndimage.label(binary)
    points = cupy_ndimage.center_of_mass(hmap, label, cp.arange(1, n_obj+1))
    #breakpoint()
    if len(points) == 0:
        return {'points': []}
    points = cp.stack(points)
    points = points.get()
    return {"points": points}

def hmap_to_points_pooling(
        hmap: np.ndarray,
        threshold:np.ndarray,
        ksize: int = 5,
        wbf_radius_multiplier: float = 0.0,
    ):
    """
    hmap: np.array with shape (D, H, W)
    threshold: float
    """
    hmap = hmap[None, None]

    pooled = F.max_pool3d(hmap, kernel_size=ksize, stride=1, padding=ksize//2)
    maxima = pooled == hmap
    maxima = maxima & (hmap > threshold)
    coords = torch.nonzero(maxima.squeeze())
    #TODO: possibly implement non-maximum suppression
    coords = coords.cpu().numpy().astype('float32')
    confidences = hmap.squeeze()[maxima.squeeze()].cpu().numpy()
    return {"points" : coords, "confidence" : confidences}

#from scipy.spatial import cKDTree as KDTree
def weighted_box_fusion(preds, particle_radius, min_votes=1):
    """
    preds: {
        points: np.ndarray with shape (n, 3)
        confidence: np.ndarray with shape (n,)
    }
    particle_radius: float
        points that are closer than this distance will be fused
    """
    new_preds = {
        'points': [],
        'confidence': [],
    }
    # sort the predections by confidence
    sorted_confidence_idx = np.argsort(preds['confidence'])[::-1]

    fused = np.zeros(len(preds['points']), dtype=bool)
    for idx in sorted_confidence_idx:
        if fused[idx]:
            continue
        dist = distance(preds['points'][idx], preds['points'][~fused])
        fusable = dist < particle_radius
        #print(np.sum(fusable))
        fusable_idx = np.where(~fused)[0][fusable]

        if len(fusable_idx) == 1:
            new_preds['points'].append(preds['points'][idx])
            new_preds['confidence'].append(preds['confidence'][idx])
            continue

        # if the number of votes is less than the minimum, ignore the point
        if len(fusable_idx) < min_votes:
            continue

        #print(f'fusable_idx: {fusable_idx}')
        new_preds['points'].append(np.average(
            preds['points'][fusable_idx],
            axis=0,
            weights=preds['confidence'][fusable_idx]
        ))
        new_preds['confidence'].append(preds['confidence'][fusable_idx].mean())
        fused[fusable_idx] = True
    new_preds['points'] = np.stack(new_preds['points'])
    new_preds['confidence'] = np.array(new_preds['confidence'])
    return new_preds

class PostProcessor:
    """
    CoPick-aware post-processing pipeline for cryo-ET particle detection inference.
    
    Automatically extracts configuration from CoPick root including:
    - Particle classes and radii
    - Tomogram dimensions per experiment
    - Voxel spacing information
    - Number of tiles per experiment based on sliding window parameters
    
    Inference pipeline:
    1. aggregate tiled predictions (accumulate)
    2. detect blobs in the accumulated heatmap (process) 
    3. calculate the centroid of the detected blobs (process)
    """
    def __init__(self,
            copick_root,
            window_size = (128, 128, 128),
            window_stride = (64, 64, 64),
            voxel_spacing: Optional[float] = None,
            threshold: Union[float, List[float]] = 0.5,
            erosion_after_threshold: Union[int, list[int]] = 0,
            weight_center: bool = False,
            edge_begin: float = 0.25,
            min_weight: float = 0.3333,
            ignore_uncovered: bool = False,
            gaussian_blur_size: int = 0,
            gaussian_blur_sigma: float = 1,
            method: str = "pooling",
            method_args: dict = {},
            wbf_radius_multiplier: float = 0.0,
            include_edge_windows: bool = True,
            device: str = 'cuda'
        ):
        """
        copick_root: CoPick root object containing particle specifications
        window_size: Size of sliding window (D, H, W)
        window_stride: Stride for sliding window (D, H, W) 
        voxel_spacing: Voxel spacing to use. If None, uses first available voxel spacing
        """
        self.copick_root = copick_root
        self.window_size = window_size
        self.window_stride = window_stride
        self.include_edge_windows = include_edge_windows
        
        # Extract classes and particle radius from CoPick configuration
        self.classes = [p.name for p in self.copick_root.pickable_objects if p.is_particle]
        self.particle_radius = {p.name: p.radius for p in self.copick_root.pickable_objects if p.is_particle}
        
        # Auto-detect voxel spacing if not provided
        if voxel_spacing is None:
            # Get first available voxel spacing from first run
            for run in self.copick_root.runs:
                if run.voxel_spacings:
                    self.voxel_spacing = list(run.voxel_spacings.keys())[0]
                    break
            else:
                raise ValueError("No voxel spacings found in any runs")
        else:
            self.voxel_spacing = voxel_spacing
            
        # Calculate tomogram shapes and tiles per experiment from copick_root
        self.experiment_info = self._analyze_experiments()
        self.tiles_per_experiment = self.experiment_info['tiles_per_experiment']
        self.pred_sizes_per_experiment = self.experiment_info['pred_sizes_per_experiment']
        
        self.accumulated_data = {}
        self.predictions = {}
        self.threshold = threshold
        self.ignore_uncovered = ignore_uncovered
        self.erosion_after_threshold = erosion_after_threshold
        if weight_center:
            self.pred_weights = create_weight(window_size, edge_begin = edge_begin, min_weight = min_weight)
        else:
            self.pred_weights = np.ones(window_size)#torch.ones(window_size, device='cuda', dtype=torch.float16)
        #self.pred_weights = 1 if not weight_center else create_weight(window_size, edge_begin = 0.25, min_weight = 0.3333)
        self.pred_weights = torch.Tensor(self.pred_weights).to(device)

        self.batch_size = 1
        self.to_process = []

        if isinstance(gaussian_blur_sigma, (int, float)):
            gaussian_blur_sigma = [gaussian_blur_sigma] * len(self.classes)
        if isinstance(gaussian_blur_size, (int, float)):
            gaussian_blur_size = [gaussian_blur_size] * len(self.classes)

        assert len(gaussian_blur_sigma) == len(self.classes), "The length of gaussian_blur_sigma should be the same as the number of classes"
        assert len(gaussian_blur_size) == len(self.classes), "The length of gaussian_blur_size should be the same as the number of classes"

            
        self.gaussian_kernels = {}
        for i, c in enumerate(self.classes):
            if gaussian_blur_size[i] == 0:
                self.gaussian_kernels[c] = None
                continue
            kernel = torch.Tensor(gaussian_kernel(gaussian_blur_sigma[i], [gaussian_blur_size[i]]*3)).to(device)
            kernel = kernel / kernel.sum()
            self.gaussian_kernels[c] = kernel[None, None]

        self.postprocess_method = method

        self.postprocess_methods = {	
            "ccl": hmap_to_points_ccl,
            "pooling": hmap_to_points_pooling,
            "ccl_gpu": hmap_to_points_ccl_gpu,
        }

        self.method_args = method_args
        self.wbf_radius_multiplier = wbf_radius_multiplier

    def _analyze_experiments(self):
        """
        Analyze all experiments in copick_root to determine tomogram shapes and calculate tiles per experiment
        """
        experiment_shapes = {}
        max_shape = [0, 0, 0]
        
        for run in self.copick_root.runs:
            voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
            if not voxel_spacing_obj or not voxel_spacing_obj.tomograms:
                print(f"No tomograms found for run {run.name} at voxel spacing {self.voxel_spacing}")
                continue
                
            tomogram = voxel_spacing_obj.tomograms[0]
            zarr_path = tomogram.zarr()
            try:
                zarr_array = zarr.open(zarr_path, 'r')['0']
                tomo_shape = zarr_array.shape  # (D, H, W)
                experiment_shapes[run.name] = tomo_shape
                
                # Update max shape for determining pred_size
                for i in range(3):
                    max_shape[i] = max(max_shape[i], tomo_shape[i])
                    
            except Exception as e:
                print(f"Error opening zarr array {zarr_path} for run {run.name}: {e}")
                continue
        
        if not experiment_shapes:
            raise ValueError("No valid tomograms found in copick_root")
            
        # Calculate tiles per experiment based on sliding window approach
        tiles_per_exp = {}
        for exp_id, tomo_shape in experiment_shapes.items():
            depth, height, width = tomo_shape
            stride_d, stride_h, stride_w = self.window_stride
            size_d, size_h, size_w = self.window_size
            
            if self.include_edge_windows:
                num_d = ceil((depth - size_d) / stride_d) + 1 if depth > size_d else 1
                num_h = ceil((height - size_h) / stride_h) + 1 if height > size_h else 1
                num_w = ceil((width - size_w) / stride_w) + 1 if width > size_w else 1
            else:
                num_d = max(1, (depth - size_d) // stride_d + 1) if depth >= size_d else 0
                num_h = max(1, (height - size_h) // stride_h + 1) if height >= size_h else 0
                num_w = max(1, (width - size_w) // stride_w + 1) if width >= size_w else 0
                
            tiles_per_exp[exp_id] = num_d * num_h * num_w
            
        return {
            'experiment_shapes': experiment_shapes,
            'tiles_per_experiment': tiles_per_exp,
            'pred_sizes_per_experiment': experiment_shapes.copy(),  # Each experiment uses its own shape
            'max_pred_size': tuple(max_shape),
            'uniform_tiles_per_exp': len(set(tiles_per_exp.values())) == 1
        }
    
    def get_experiment_ids(self):
        """Get list of experiment IDs available in copick_root"""
        return [run.name for run in self.copick_root.runs]
    
    def get_available_voxel_spacings(self):
        """Get list of available voxel spacings across all runs"""
        spacings = set()
        for run in self.copick_root.runs:
            spacings.update(run.voxel_spacings.keys())
        return sorted(list(spacings))
    
    def get_tomogram_shape(self, experiment_id: str):
        """Get tomogram shape for a specific experiment"""
        return self.experiment_info['experiment_shapes'].get(experiment_id)
    
    def get_tiles_count(self, experiment_id: str):
        """Get expected number of tiles for a specific experiment"""
        return self.tiles_per_experiment.get(experiment_id)
    
    def get_pred_size(self, experiment_id: str):
        """Get prediction size for a specific experiment"""
        if experiment_id in self.pred_sizes_per_experiment:
            return self.pred_sizes_per_experiment[experiment_id]
        else:
            # For unknown experiments, use a default size that will be dynamically determined
            return None

    def process(self, heatmap: np.ndarray, threshold: Union[float, List[float]] = 0.5, erosion: Union[int, List[int]] = 0): 
        """
        performs a blob detection on the accumulated heatmap, then returns the center coordinates of the detected blobs
        heatmap should have a shape of (n_classes, D, H, W)
        offset: np.ndarray with shape (3, D, H, W)
        """
        class_points = {}
        for i, c in enumerate(self.classes):
            #points = hmap_to_points_ccl(heatmap[i], threshold if isinstance(threshold, float) else threshold[i])
            if self.gaussian_kernels[c] is not None:
                heatmap[i] = F.conv3d(heatmap[i][None, None], self.gaussian_kernels[c], padding=self.gaussian_kernels[c].shape[-1]//2)[0, 0]

            preds = self.postprocess_methods[self.postprocess_method](
                heatmap[i],
                threshold if isinstance(threshold, float) else threshold[i],
                **self.method_args
            )

            if len(preds["points"]) == 0:
                class_points[c] = {
                    'points': np.zeros((0, 3)),
                    'confidence': np.zeros(0),
                }
                continue

            wbf_radius_mul = self.wbf_radius_multiplier if isinstance(self.wbf_radius_multiplier, (int, float)) else self.wbf_radius_multiplier[i]
            if wbf_radius_mul > 0:
                if not "confidence" in preds:
                    print("Warning: confidence is not available for the weighted box fusion, setting all confidences to 1")
                    preds["confidence"] = np.ones(len(preds["points"]))
                preds = weighted_box_fusion(preds, self.particle_radius[c]*wbf_radius_mul)
            points = preds["points"]
            points = points[:, ::-1]# z, y, x -> x, y, z
            points *= self.voxel_spacing
            class_points[c] = {
                'points': points,
            }
            if 'confidence' in preds:
                class_points[c]['confidence'] = preds['confidence']
        return class_points

    def reset(self):
        self.accumulated_data = {}
        self.predictions = {}
    
    def accumulate(self, heatmaps: np.ndarray, crop_origins: np.ndarray, experiment_ids: List[str]):
        """
        accumulates the data for the past frames
        samples with new experiment_id should only be passed after all the samples with the previous experiment_id have been passed
        """
        for i, experiment_id in enumerate(experiment_ids):
            heatmap = heatmaps[i]
            crop_size = heatmap.shape[1:]
            crop_origin = crop_origins[i]
            
            if experiment_id not in self.accumulated_data:
                # Get prediction size for this experiment
                pred_size = self.get_pred_size(experiment_id)
                if pred_size is None:
                    # For unknown experiments, calculate required size from crop
                    pred_size = tuple(crop_origin + crop_size)
                    # Store the dynamically determined size
                    self.pred_sizes_per_experiment[experiment_id] = pred_size
                
                self.accumulated_data[experiment_id] = {
                    'heatmaps': torch.zeros((len(self.classes), *pred_size), device=heatmaps.device, dtype=torch.float16),
                    'count': torch.zeros(pred_size, device=heatmaps.device, dtype=torch.float16),
                    'n_tiles': 0,
                }

            self.accumulated_data[experiment_id]['heatmaps'][
                :,
                crop_origin[0]: crop_origin[0] + crop_size[0],
                crop_origin[1]: crop_origin[1] + crop_size[1],
                crop_origin[2]: crop_origin[2] + crop_size[2],
            ] += heatmap * self.pred_weights
            self.accumulated_data[experiment_id]['count'][
                crop_origin[0]: crop_origin[0] + crop_size[0],
                crop_origin[1]: crop_origin[1] + crop_size[1],
                crop_origin[2]: crop_origin[2] + crop_size[2],
            ] += self.pred_weights
            self.accumulated_data[experiment_id]['n_tiles'] += 1

            # Get experiment-specific tile count
            expected_tiles = self.tiles_per_experiment.get(experiment_id, 1)  # Default to 1 tile for unknown experiments
            if self.accumulated_data[experiment_id]['n_tiles'] == expected_tiles:
                self.predictions[experiment_id] = {}
                #if self.batch_size > 1:
                #    self.to_process.append(experiment_id)
                if self.accumulated_data[experiment_id]['count'].min() == 0 and not self.ignore_uncovered:
                    raise ValueError('Some regions are not covered by the tiles')
                heatmap = self.accumulated_data[experiment_id]['heatmaps'] / self.accumulated_data[experiment_id]['count']
                    
                self.predictions[experiment_id].update(self.process(heatmap, threshold = self.threshold))

                del self.accumulated_data[experiment_id]
                