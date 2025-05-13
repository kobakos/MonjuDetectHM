from typing import Tuple, List, Union, Optional, Dict

import numpy as np

def generate_gaussian_kernel(size, sigma, cutoff_radius=3):
    """
    size:
        array of int, size of the kernel. should be odd
    sigma:
        float, standard deviation of the kernel
    """
    indices = np.indices(size)
    indices = indices.reshape(len(size), -1)
    indices -= np.asarray(size)[:, None] // 2
    kernel = np.exp(-0.5 * np.sum(indices ** 2, axis=0) / sigma ** 2)
    return indices, kernel

def generate_offset_kernel(radius: int):
    """
    radius:
        int, radius of the kernel
    """
    size = [2 * radius + 1] * 3
    indices = np.indices(size)
    indices = indices.reshape(len(size), -1)
    indices -= np.asarray(size)[:, None] // 2
    dist = np.linalg.norm(indices, axis=0)
    inside = dist <= radius
    indices = indices[:,inside]
    kernel = indices / radius
    return indices, kernel

def crop_points(points: np.ndarray, crop: np.ndarray, margin: int=10, reset_origin: bool=True):
    """
    points:
        numpy array with shape (n_points, 3). The last dimension should correspond to the order of the axis (NOT x, y, z)
    crop:
        array with shape (3, 2), where the first dimension corresponds to the axis and the second to the start and end of the crop
    returns points that are within the crop, with a specified margin
    """
    #pad_front = np.maximum(-crop[:, 0], 0)

    # currently does not support filling the padded area with points
    #points += pad_front[None, :]
    #crop += pad_front[:, None]

    points = points[((points > (crop[:, 0] - margin)) & (points < (crop[:, 1] + margin))).all(axis=-1), :]
    if reset_origin:
        points = points - crop[:, 0][None, :]
    return points

class TargetGenerator():
    def __init__(self,
            target_size: Tuple,
            classes: List[str],
            particle_radius: Dict[str, int],
            resolution_hierarchy: int = 0,
            crop_margin: int = 10,
            kernel_size_multiplier: int = 3,
            kernel_sigma_multiplier: float = 1/3,
            return_as_points: bool = False,
            return_num_points: bool = False,
            generate_offset_target: bool = False,
            offset_radius_multiplier: float = 0.1,
        ):
        self.target_size = np.asarray(target_size)
        self.classes = classes
        self.particle_radius = particle_radius
        self.crop_margin = crop_margin
        self.pixel_spacing = [
            10.012444196428572, 
            10.012444196428572*2,
            10.012444196428572*4][resolution_hierarchy]
        self.return_as_points = return_as_points
        self.return_num_points = return_num_points

        self.gaussian_kernels = {
            cls: generate_gaussian_kernel(
                [particle_radius[cls]*kernel_size_multiplier]*3,
                particle_radius[cls] * (kernel_sigma_multiplier if isinstance(kernel_sigma_multiplier, (int, float)) else kernel_sigma_multiplier[i])
            )
            for i, cls in enumerate(classes)
        }

        self.generate_offset_target = generate_offset_target
        self.offset_kernels = {
            cls: generate_offset_kernel(round(particle_radius[cls] * offset_radius_multiplier))
            for cls in classes
        }

    def __call__(self, class_points_angstrom: Dict[str, np.ndarray], crop: Optional[np.ndarray]=None, affine: Optional[np.ndarray]=None):
        """
        crop should be none or array with shape (3, 2), where the first dimension corresponds to the axis and the second to the start and end of the crop
        """
        total_points = 0
        #target = [] if not self.return_as_points else {'heatmaps': [], 'labels': [], 'boxes': [], 'num_points': 0}# it is named boxes to reuse the code of detr
        target = {
            'heatmap': np.zeros((len(self.classes), *self.target_size), dtype=np.float32),
            'points': {c: [] for c in self.classes},
        }
        if self.generate_offset_target:
            target['offset'] = np.full((len(self.classes), 3, *self.target_size), -1, dtype=np.float32)
        for class_index, class_name in enumerate(self.classes):
            points_angstrom = class_points_angstrom[class_name]
            points = self.convert_to_pixels(points_angstrom)
            if crop is not None:
                points = crop_points(points, crop, margin = self.crop_margin)
            if affine is not None and len(points) > 0:
                points = (affine[:3, :3] @ points.T + affine[:3, 3:]).T
                points = crop_points(points, np.array([[0, self.target_size[0]], [0, self.target_size[1]], [0, self.target_size[2]]]), margin = self.crop_margin, reset_origin=False)
            if self.return_num_points:
                num_points = len(points)
                total_points += num_points
            target['points'][class_name] = points
            if self.return_as_points:
                target['labels'] = target.get('labels', []) + [class_index] * len(points)
                if 'boxes' not in target:
                    target['boxes'] = []
                target['boxes'].append(points)
            else:
                self.generate_single_class(points, class_name, target['heatmap'][class_index])
                if self.generate_offset_target:
                    self.generate_offset_target_single_class(points, class_name, target['offset'][class_index])
                    
                
        if self.return_num_points:            
            target['num_points'] = total_points
        if self.return_as_points:
            target['labes'] = np.array(target['labels'])
            target['boxes'] = np.concatenate(target['boxes'], axis=0)
            target['boxes'] = target['boxes'] / self.target_size[None, :]# normalize the coords
            #target['labels'] = np.eye(len(self.classes))[target['labels']]# one hot encode the labels
        return target
    
    def generate_offset_target_single_class(self, points: np.ndarray, class_name: str, target: np.ndarray):
        # target: (3, *self.target_size)
        for point in points:
            point = np.round(point).astype(int)
            self.add_offsets(target, point, class_name)

    def add_offsets(self, target: np.ndarray, point: np.ndarray, class_name: str):
        indices, kernel = self.offset_kernels[class_name]
        indices = indices + point[:, None]
        out_of_bound_mask = ((indices < 0) | (indices >= self.target_size[:, None])).any(axis=0)
        indices = indices[:, ~out_of_bound_mask]
        kernel = kernel[:, ~out_of_bound_mask]
        channel_idx = np.arange(3)[:, None].repeat(indices.shape[1], axis=1)
        np.minimum.at(
            target,
            (
                channel_idx,
                indices[0],
                indices[1],
                indices[2],
            ),
            kernel
        )
    
    def generate_single_class(self, points: np.ndarray, class_name: str, target: np.ndarray):
        for point in points:
            point = np.round(point).astype(int)
            self.add_gaussian(target, point, class_name)

    def add_gaussian(self, target: np.ndarray, point: np.ndarray, class_name: str):
        indices, kernel = self.gaussian_kernels[class_name]
        indices = indices + point[:, None]
        out_of_bound_mask = ((indices < 0) | (indices >= self.target_size[:, None])).any(axis=0)
        indices = indices[:, ~out_of_bound_mask]
        kernel = kernel[~out_of_bound_mask]
        np.maximum.at(
            target,
            (
                indices[0],
                indices[1],
                indices[2],
            ),
            kernel
        )
    
    def convert_to_pixels(self, points: np.ndarray):
        return points / self.pixel_spacing # the spacing

    
if __name__ == '__main__':
    target_generator = TargetGenerator(
        target_size=(128, 128, 128),
        classes=['class1', 'class2'],
        particle_radius_angstrom={'class1': 5, 'class2': 10},
        resolution_hierarchy=0
    )
    class_points_angstrom = {
        'class1': np.random.rand(10, 3) * 100,
        'class2': np.random.rand(10, 3) * 100
    }
    import time
    times = []
    for _ in range(1000):
        start = time.time()
        target = target_generator(class_points_angstrom)
        end = time.time()
        times.append(end - start)
    print(np.mean(times), np.std(times))
