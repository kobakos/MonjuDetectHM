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
            particle_radius: Dict[str, float],
            voxel_spacing: float = 10.0,
            crop_margin: int = 10,
            kernel_size_multiplier: int = 3,
            kernel_sigma_multiplier: float = 1/3,
        ):
        self.target_size = np.asarray(target_size)
        self.classes = classes
        self.crop_margin = crop_margin

        self.voxel_spacing = voxel_spacing

        particle_radius_pix={cls: int(radius / voxel_spacing) for cls, radius in particle_radius.items()}

        self.gaussian_kernels = {
            cls: generate_gaussian_kernel(
                [int(particle_radius_pix[cls]*kernel_size_multiplier)]*3,
                particle_radius_pix[cls] * (kernel_sigma_multiplier if isinstance(kernel_sigma_multiplier, (int, float)) else kernel_sigma_multiplier[i])
            )
            for i, cls in enumerate(classes)
        }

    def __call__(self, class_points: Dict[str, np.ndarray], crop: Optional[np.ndarray]=None, affine: Optional[np.ndarray]=None):
        """
        crop should be none or array with shape (3, 2), where the first dimension corresponds to the axis and the second to the start and end of the crop
        """
        #target = [] if not self.return_as_points else {'heatmaps': [], 'labels': [], 'boxes': [], 'num_points': 0}# it is named boxes to reuse the code of detr
        target = {
            'heatmap': np.zeros((len(self.classes), *self.target_size), dtype=np.float32),
            'points': {},
        }
        for class_index, class_name in enumerate(self.classes):
            points = class_points[class_name]
            #points = self.to_pixels(points_angstrom)
            if crop is not None:
                points = crop_points(points, crop, margin = self.crop_margin)
            if affine is not None and len(points) > 0:
                points = (affine[:3, :3] @ points.T + affine[:3, 3:]).T
                points = crop_points(points, np.array([[0, self.target_size[0]], [0, self.target_size[1]], [0, self.target_size[2]]]), margin = self.crop_margin, reset_origin=False)
            target['points'][class_name] = points
            self.generate_single_class(points, class_name, target['heatmap'][class_index])
        return target
    
    def to_pixels(self, points):
        """
        Converts points in Angstrom to pixels based on the target size and voxel spacing.
        """
        return points / self.voxel_spacing
    
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
