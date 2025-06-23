from typing import Tuple, Union, Optional

import numpy as np
from scipy.ndimage import affine_transform

class InputPreprocessor():
    def __init__(self,
            image_size: Tuple[int, int, int] = (128, 128, 128),
            rescale_factor: float = 1e5, clip_percentile: Tuple[float, float] = (0.1, 99.9), standardize: bool = False, pad_mode: str = 'constant'
        ):
        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.clip_percentile = clip_percentile
        self.standardize = standardize
        self.pad_mode = pad_mode
    
    def __call__(self, image, crop = None, inv_affine = None):
        """
        image should be a zarr object with 3 resolution levels
        """
        #print(pascal_bbox.shape, 'before_crop')
        if crop is not None:
            pad_front = np.maximum(-crop[:, 0], 0)
            if (pad_front>0).any():
                assert inv_affine is not None, 'padding is applied although no affine is provided'
                inv_affine[:3, 3] = inv_affine[:3, 3] - pad_front
                crop = np.clip(crop, 0, None)
            # pad_back = np.maximum(crop[:, 1] - np.asarray(self.image_size), 0)
            # pad = np.stack([pad_front, pad_back], axis=1)
            # image = np.pad(image, pad, mode=self.pad_mode)
            # crop = crop + pad_front[:, None]

            if isinstance(image, dict):
                for k in image.keys():
                    image[k]['image'] = image[k]['image'][
                        crop[0][0]: crop[0][1],
                        crop[1][0]: crop[1][1],
                        crop[2][0]: crop[2][1],
                    ]
            else:
                image = image[
                    crop[0][0]: crop[0][1],
                    crop[1][0]: crop[1][1],
                    crop[2][0]: crop[2][1],
                ]
        if isinstance(image, dict):
            image = sum([np.asarray(image[k]['image']) * image[k]['weight'] for k in image.keys()])
        
        #start=time.time()
        image = np.asarray(image)

        if inv_affine is not None:
            #print('begin affine', time.time())
            #image = affine_transform(image, matrix=np.asarray(inv_affine), output_shape=tuple(self.image_size), order=1)
            image = affine_transform(image, matrix=inv_affine, output_shape=tuple(self.image_size), order=1)
            #print('affine done', time.time())

        image = percentile_clip(image, self.clip_percentile[0], self.clip_percentile[1])

        if self.standardize:
            image -= np.mean(image)
            image /= np.std(image)
            #image = (image - np.mean(image)) / np.std(image)

        if self.rescale_factor != 1:
            image *= self.rescale_factor

        #image = image.get()
        return image
#@jit(nopython=True)    
def percentile_clip(image, low, high):
    """
    clips the image to the specified percentiles
    """
    if low == 0 and high == 100:
        return image
    low, high = np.percentile(image, [low, high])
    image[image < low] = low
    image[image > high] = high
    #image = np.clip(image, low, high)
    return image