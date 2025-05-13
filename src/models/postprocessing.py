from typing import List, Union, Tuple, Optional
import concurrent.futures

import torch
import torch.nn.functional as F
import numpy as np

from scipy import ndimage

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
    Inference pipeline:
    1. aggregate tiled predictions (accumulate)
    2. detect blobs in the accumulated heatmap (process)
    3. calculate the centroid of the detected blobs (process)
    """
    def __init__(self,
            classes: List[str],
            tiles_per_experiment: int=162,
            pred_size = (184, 630, 630),
            window_size = (128, 128, 128),
            resolution_hierarchy = 0,
            threshold: Union[float, List[float]] = 0.5,
            erosion_after_threshold: Union[int, list[int]] = 0,
            weight_center: bool = False,
            edge_begin: float = 0.25,
            min_weight: float = 0.3333,
            ignore_uncovered: bool = False,
            gaussian_blur_size: int = 0,
            gaussian_blur_sigma: float = 1,
            method: str = "ccl",
            method_args: dict = {},
            wbf_radius_multiplier: float = 0.0,
            keep_heatmaps: bool = False,
            keep_accumulated: bool = False,
        ):
        """
        classes: list of class names. should be in the same order as the channel dimension of the predicted heatmap
        """
        self.accumulated_data = {}# {experiment_id: {'heatmap': np.array(n_classes, *pred_size), 'count': np.array(*pred_size), 'offset': array(3, *pred_size)} }
        self.classes = classes
        self.tiles_per_experiment = tiles_per_experiment
        self.predictions = {}# {experiment_id: {class: [[x, y, z], ...], class_conf: [float, ...]}}
        self.pred_size = pred_size
        self.threshold = threshold
        self.pixel_spacing = [
            10.012444196428572, 
            10.012444196428572*2,
            10.012444196428572*4][resolution_hierarchy]
        self.ignore_uncovered = ignore_uncovered
        self.erosion_after_threshold = erosion_after_threshold
        if weight_center:
            self.pred_weights = create_weight(window_size, edge_begin = edge_begin, min_weight = min_weight)
        else:
            self.pred_weights = np.ones(window_size)#torch.ones(window_size, device='cuda', dtype=torch.float16)
        #self.pred_weights = 1 if not weight_center else create_weight(window_size, edge_begin = 0.25, min_weight = 0.3333)
        self.pred_weights = torch.Tensor(self.pred_weights).cuda()
        
        self.particle_radius = {
            'apo-ferritin': 60,  # apo-ferritin
            'beta-amylase': 65,  # beta-amylase
            'beta-galactosidase': 90,  # beta-galactosidase
            'ribosome': 150,  # ribosome
            'thyroglobulin': 130,  # thyroglobulin
            'virus-like-particle': 135,  # virus-like-particle
        }

        self.batch_size = 1
        self.to_process = []

        if isinstance(gaussian_blur_sigma, (int, float)):
            gaussian_blur_sigma = [gaussian_blur_sigma] * len(classes)
        if isinstance(gaussian_blur_size, (int, float)):
            gaussian_blur_size = [gaussian_blur_size] * len(classes)

        assert len(gaussian_blur_sigma) == len(classes), "The length of gaussian_blur_sigma should be the same as the number of classes"
        assert len(gaussian_blur_size) == len(classes), "The length of gaussian_blur_size should be the same as the number of classes"

            
        self.gaussian_kernels = {}
        for i, c in enumerate(classes):
            if gaussian_blur_size[i] == 0:
                self.gaussian_kernels[c] = None
                continue
            kernel = torch.Tensor(gaussian_kernel(gaussian_blur_sigma[i], [gaussian_blur_size[i]]*3)).cuda().half()
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

        self.keep_heatmaps=keep_heatmaps
        self.keep_accumulated=keep_accumulated

        self.adjust_th_on_the_fly = False
        self.base_n_points = {
            'apo-ferritin': 239/7,
            'beta-amylase': 0,
            'beta-galactosidase': 0,
            'ribosome': 0,
            'thyroglobulin': 0,
            'virus-like-particle': 0,
        }

    def process(self, heatmap: np.ndarray, offset: Optional[np.ndarray] = None, threshold: Union[float, List[float]] = 0.5, erosion: Union[int, List[int]] = 0): 
        """
        performs a blob detection on the accumulated heatmap, then returns the center coordinates of the detected blobs
        heatmap should have a shape of (n_classes, D, H, W)
        offset: np.ndarray with shape (3, D, H, W)
        """
        class_points = {}
        # {
        #     class_str: {
        #         'points': np.ndarray with shape (n, 3)
        #         'confidence': np.ndarray with shape (n,)
        #     }
        # }
        for i, c in enumerate(self.classes):
            #points = hmap_to_points_ccl(heatmap[i], threshold if isinstance(threshold, float) else threshold[i])
            if self.gaussian_kernels[c] is not None:
                heatmap[i] = F.conv3d(heatmap[i][None, None], self.gaussian_kernels[c], padding=self.gaussian_kernels[c].shape[-1]//2)[0, 0]

            preds = self.postprocess_methods[self.postprocess_method](
                heatmap[i],
                threshold if isinstance(threshold, float) else threshold[i],
                **self.method_args
            )

            if self.adjust_th_on_the_fly:
                n_points = len(preds['points'])
                new_th = threshold + (n_points - self.base_n_points[c]) * 1e-3
                new_th = np.clip(new_th, -0.3, 1.0)
                preds = self.postprocess_methods[self.postprocess_method](
                    heatmap[i],
                    new_th,
                    **self.method_args
                )


            if len(preds["points"]) == 0:
                class_points[c] = {
                    'points': np.zeros((0, 3)),
                    'confidence': np.zeros(0),
                }
                continue

            if offset is not None:
                preds["points"] += offset[
                    :, 
                    preds["points"][:, 0], 
                    preds["points"][:, 1], 
                    preds["points"][:, 2]
                ].cpu().numpy().T * (self.particle_radius[c] / self.pixel_spacing)

            wbf_radius_mul = self.wbf_radius_multiplier if isinstance(self.wbf_radius_multiplier, (int, float)) else self.wbf_radius_multiplier[i]
            if wbf_radius_mul > 0:
                if not "confidence" in preds:
                    print("Warning: confidence is not available for the weighted box fusion, setting all confidences to 1")
                    preds["confidence"] = np.ones(len(preds["points"]))
                preds = weighted_box_fusion(preds, self.particle_radius[c]*wbf_radius_mul)
            points = preds["points"]
            points = points[:, ::-1]# z, y, x -> x, y, z
            points *= self.pixel_spacing
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
        if isinstance(heatmaps, (list, tuple)):
            heatmaps, offsets = heatmaps
        else:
            offsets = None
        heatmaps = heatmaps.half()
        for i, experiment_id in enumerate(experiment_ids):
            if experiment_id not in self.accumulated_data:
                self.accumulated_data[experiment_id] = {
                    'heatmaps': torch.zeros((len(self.classes), *self.pred_size), device=heatmaps.device, dtype=torch.float16),
                    'count': torch.zeros(self.pred_size, device=heatmaps.device, dtype=torch.float16),
                    'n_tiles': 0,
                }
                if offsets is not None:
                    self.accumulated_data[experiment_id]['offset'] = torch.zeros((3, *self.pred_size), device=heatmaps.device, dtype=torch.float16)

            heatmap = heatmaps[i]
            crop_size = heatmap.shape[1:]
            crop_origin = crop_origins[i]
            offset = offsets[i] if offsets is not None else None

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
            if offset is not None:
                self.accumulated_data[experiment_id]['offset'][
                    :,
                    crop_origin[0]: crop_origin[0] + crop_size[0],
                    crop_origin[1]: crop_origin[1] + crop_size[1],
                    crop_origin[2]: crop_origin[2] + crop_size[2],
                ] += offset
            self.accumulated_data[experiment_id]['n_tiles'] += 1

            if self.accumulated_data[experiment_id]['n_tiles'] == self.tiles_per_experiment:
                self.predictions[experiment_id] = {}
                #if self.batch_size > 1:
                #    self.to_process.append(experiment_id)
                if self.accumulated_data[experiment_id]['count'].min() == 0 and not self.ignore_uncovered:
                    print(self.accumulated_data[experiment_id]['count'])
                    print((self.accumulated_data[experiment_id]['count']==0).sum())
                    print(f"Experiment {experiment_id} has uncovered regions")
                    raise ValueError('Some regions are not covered by the tiles')
                heatmap = self.accumulated_data[experiment_id]['heatmaps'] / self.accumulated_data[experiment_id]['count']
                if 'offset' in self.accumulated_data[experiment_id]:
                    offset = self.accumulated_data[experiment_id]['offset'] / self.accumulated_data[experiment_id]['count']
                else:
                    offset = None

                if self.keep_heatmaps:
                    self.predictions[experiment_id]['heatmap'] = heatmap.detach().cpu().numpy()
                    
                self.predictions[experiment_id].update(self.process(heatmap, offset=offset, threshold = self.threshold))

                del self.accumulated_data[experiment_id]
                
        def process_remaining(self):
            self.process_batch(self.to_process)

        def process_batch(self, heatmaps: List[np.ndarray], threshold: float, mode = "ccl"):
            """
            performs a blob centroid detection of heatmaps in batch
            heatmaps should be a list of np.array with shape (n_classes, D, H, W)
            threshold should be the detection threshold, which its meaning may differ among the modes
            mode should be either "ccl" or "pooling"
            """
            assert mode in ["ccl", "pooling"], f"Unsupported mode: {mode}"
            if mode == "ccl":
                self.process_batch_ccl(heatmaps, threshold)
            elif mode == "pooling":
                self.process_batch_pooling(heatmaps, threshold)

        def process_batch_ccl(self, heatmaps: List[np.ndarray], threshold: float):
            """
            performs a blob centroid detection of heatmaps in batch using connected component labeling
            heatmaps should be a list of np.array with shape (n_classes, D, H, W)
            threshold should be the detection threshold
            """
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(
                    process_single_hmap_ccl,
                    None,#bbox_points
                ) for c in self.classes for hmap in heatmaps]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()

                    if return_metadata is not None and 'metadata' in result:
                        metadata['positions'].append(result['metadata']['position'])
                        metadata['instance_numbers'].append(result['metadata']['instance_number'])
                        if metadata['orientation'] is None:
                            metadata['orientation'] = result['metadata']['orientation']
                            metadata['spacing'] = result['metadata']['spacing']

                    if result['coord_mm'] is not None:
                        for coord_mm, disc_class, condition_class in result['coord_mm']:
                            coords[disc_class][condition_class] = coord_mm
        def process_single_hmap_ccl(hmap: np.ndarray, threshold: float):
            """
            performs a blob centroid detection of a single heatmap using connected component labeling
            hmap should have a shape of (D, H, W)
            threshold should be the detection threshold
            """
            binary = hmap > threshold
            label, n_obj = ndimage.label(binary)
            points = ndimage.center_of_mass(hmap, label)


if __name__ == '__main__':
    import pprint
    import matplotlib.pyplot as plt
    # test find_connected_component
    ind = np.indices((128, 128, 128))
    hmap = np.zeros((5, 128, 128, 128))
    points = {i: [] for i in range(5)}
    for c in range(5):
        for _ in range(np.random.randint(5, 10)):
            point = np.random.randint(0, 128, 3)
            points[c].append(point)
            hmap[c] = np.maximum(
                gaussian_kernel(ind, point, 10),
                hmap[c]
            )
    
    postprocessor = PostProcessor(['class1', 'class2', 'class3', 'class4', 'class5'])
    print(postprocessor.process(hmap, 0))


    threshold = [0.5]*5
    max_radius = 100
    pprint.pprint(points[0])
    print(hmap_to_points_ccl(hmap[0], 0.5))
    exit()
    hmap = torch.Tensor(hmap).cuda()
    component = find_connecte_component(hmap, threshold, max_radius)
    print(component.shape)
    for i in range(0, 128, 10):
        plt.imshow(component[0, i].cpu().numpy())
        plt.show()
    # test find_centroid
    centroid = find_centroid(component)
    pprint.pprint(points)
    print(centroid)
    breakpoint()
    # test PastProcessor
    classes = ['class1', 'class2']
    pred_size = (10, 10, 10)
    past_processor = PostProcessor(classes, pred_size)
    heatmaps = np.random.rand(2, 10, 10, 10)
    crop_origins = np.random.randint(0, 10, (2, 3))
    experiment_ids = [0, 0]
    past_processor.accumulate(heatmaps, crop_origins, experiment_ids)
    print(past_processor.accumulated_data)