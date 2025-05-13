import numpy as np


def calculate_required_shape(
        inv_linear_transform: np.ndarray,
        target_shape: np.ndarray,
    ):
    """
    linear_transform: np.ndarray (3, 3)
    target_shape: np.ndarray (d0, d1, d2)
    """
    # two of these is enough since the z axis is not rotated
    test_vectors = np.array([
        [target_shape[0] / 2, target_shape[1] / 2, target_shape[2] / 2],
        [target_shape[0] / 2, target_shape[1] / 2, -target_shape[2] / 2],
    ]).T
    transformed_vectors = inv_linear_transform @ test_vectors
    required_shape = np.max(np.abs(transformed_vectors), axis=1) * 2
    return np.ceil(required_shape).astype(int)

def append_translation(
        inv_linear_transform: np.ndarray,
        target_shape: np.ndarray,
        required_shape: np.ndarray
    ):
    """
    inv_linear_transform: np.ndarray (3, 3)
    target_shape: np.ndarray (d0, d1, d2)
    required_shape: np.ndarray (d0, d1, d2)

    center of the target shape should be transformed to the center of the required shape
    """
    # check where the center of the target shape is transformed to
    target_center = np.array([target_shape[0] / 2, target_shape[1] / 2, target_shape[2] / 2])
    transformed_center = inv_linear_transform @ target_center
    # move the center of the required shape to the transformed center
    delta = required_shape / 2 - transformed_center
    inv_mat = np.block([
        [inv_linear_transform, delta[:, None]],
        [np.zeros(3), 1]
    ])
    
    return inv_mat

def generate_affine_mats(
        target_shape: np.ndarray,
        rot1: float,
        scales: np.ndarray,
        rot2: float
    ):
    """
    target_shape: np.ndarray (d0, d1, d2)
    rot1: float
    scales: np.ndarray (d0, d1, d2)
    rot2: float

    output:
        tr_mat: np.ndarray (4, 4)
            affine matrix
        inv_tr_mat: np.ndarray (4, 4)
            inverse of the affine matrix
        required_shape: np.ndarray (d0, d1, d2)
            shape of the original image required to fit the target image
    This function uses a decomposition similar to svd to generate
    the affine matrix. The difference is that the rotation is only
    done around the z-axis (first dimention).
    """
    inv_rot_mat_1 = np.array([
        [1, 0, 0],
        [0, np.cos(rot1), np.sin(rot1)],
        [0, -np.sin(rot1), np.cos(rot1)],
    ])
    inv_scale_mat = np.diag(1 / scales)
    inv_rot_mat_2 = np.array([
        [1, 0, 0],
        [0, np.cos(rot2), np.sin(rot2)],
        [0, -np.sin(rot2), np.cos(rot2)],
    ])
    inv_tr_mat_wo_translation = inv_rot_mat_2 @ inv_scale_mat @ inv_rot_mat_1
    required_shape = calculate_required_shape(inv_tr_mat_wo_translation, target_shape)
    inv_mat = append_translation(inv_tr_mat_wo_translation, target_shape, required_shape)
    #mat = append_translation(np.linalg.inv(inv_tr_mat_wo_translation), required_shape, target_shape)
    mat = np.linalg.inv(inv_mat)
    #print(mat, np.linalg.inv(inv_mat))
    return mat, inv_mat, required_shape

class Augmentator:
    def __init__(self, args: dict, probabilities: dict, image_size = (128, 128, 128)):
        self.a = args
        self.p = probabilities
        self.image_size = image_size
    def __call__(self, image, target, applied_augmentations={}):
        # target_type is either 'heatmap' or 'point_dict'
        ## spacial augmentation, skip if affine is already applied
        if 'affine' not in applied_augmentations:
            if 'flip0' in self.p and np.random.rand() < self.p['flip0']:
                applied_augmentations['flip0'] = True
                image = np.flip(image, axis=0)
                if target is not None:
                    if 'heatmap' in target:
                        target["heatmap"] = np.flip(target["heatmap"], axis=1)
                    if "points" in target:
                        for c in target['points'].keys():
                            target['points'][c][:, 0] = self.image_size[0] - target['points'][c][:, 0]

            if 'flip1' in self.p and np.random.rand() < self.p['flip1']:
                applied_augmentations['flip1'] = True
                image = np.flip(image, axis=1)
                if target is not None:
                    if 'heatmap' in target:
                        target['heatmap'] = np.flip(target['heatmap'], axis=2)
                    if 'points' in target:
                        for c in target['points'].keys():
                            target['points'][c][:, 1] = self.image_size[1] - target['points'][c][:, 1]
            if 'flip2' in self.p and np.random.rand() < self.p['flip2']:
                applied_augmentations['flip2'] = True
                image = np.flip(image, axis=2)
                if target is not None:
                    if 'heatmap' in target:
                        target['heatmap'] = np.flip(target['heatmap'], axis=3)
                    if 'points' in target:
                        for c in target['points'].keys():
                            target['points'][c][:, 2] = self.image_size[2] - target['points'][c][:, 2]

            if 'rot12' in self.p and np.random.rand() < self.p['rot12']:
                k = np.random.choice([1, 2, 3])
                applied_augmentations['rot12'] = k
                image = np.rot90(image, k=k, axes=(1, 2))
                if target is not None:
                    if 'heatmap' in target:
                        target['heatmap'] = np.rot90(target['heatmap'], k=k, axes=(2, 3))
                    if 'points' in target:
                        for c in target['points'].keys():
                            target['points'][c] = self.rotate_points_12(target['points'][c], k)
        if 'brightness' in self.p and np.random.rand() < self.p['brightness']:
            brightness = np.random.uniform(self.a['brightness']['low'], self.a['brightness']['high'])
            applied_augmentations['brightness'] = brightness
            image += brightness

        if 'contrast' in self.p and np.random.rand() < self.p['contrast']:
            contrast = np.random.uniform(self.a['contrast']['low'], self.a['contrast']['high'])
            applied_augmentations['contrast'] = contrast
            image *= contrast
        
        if 'gamma' in self.p and np.random.rand() < self.p['gamma']:
            gamma = np.random.uniform(self.a['gamma']['low'], self.a['gamma']['high'])
            applied_augmentations['gamma'] = gamma
            image_sgn = np.sign(image)
            image = np.abs(image) ** gamma * image_sgn

        # wbp and ctfdeconvolved are already very noisy so we skip adding noise to them
        if 'noise' in self.p and np.random.rand() < self.p['noise']:
            intensity = np.random.uniform(self.a['noise']['low'], self.a['noise']['high'])
            applied_augmentations['noise'] = intensity
            image += np.random.normal(0, intensity, image.shape)
        
        #image = image.copy()
        image = np.ascontiguousarray(image)
        #target = target.copy() if target is not None else target
        if 'heatmap' in target:
            target['heatmap'] = np.ascontiguousarray(target['heatmap'])
        return {'image': image, 'target': target, 'applied_augmentations': applied_augmentations}
    def rotate_points_12(self, points, k):
        # k should be 0 ~ 3
        if k == 0:
            return points
        if k == 1:
            points[:, [1, 2]] = points[:, [2, 1]]
            points[:, 1] = self.image_size[1] - points[:, 1]
            return points
        if k == 2:
            points[:, 1] = self.image_size[1] - points[:, 1]
            points[:, 2] = self.image_size[2] - points[:, 2]
            return points
        if k == 3:
            points[:, [1, 2]] = points[:, [2, 1]]
            points[:, 2] = self.image_size[2] - points[:, 2]
            return points