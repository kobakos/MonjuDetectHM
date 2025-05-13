import os

import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_configs
settings = load_configs("SETTINGS.json")
BASE_PATH = settings['base_path']


def slices_visualizer(inputs, spacing = 25, cmap='gray', savedir=None):
    for i in range(0, inputs.shape[0], spacing):
        plt.imshow(inputs[i], cmap='gray')
        if savedir:
            plt.savefig(f'{savedir}/{i}.png')
            plt.clf()
        else:
            plt.show()

def visualize_intensity_distribution(zarr_path):
    arr = zarr.open(zarr_path)[0]
    arr = np.array(arr).flatten()
    plt.hist(arr, bins=100)
    plt.show()

import cv2
import random
def show_all_variants(experiment_id=None, crop=None, div=5):
    os.makedirs('./visualization_results/types', exist_ok=True)
    if experiment_id is None:
        experiment_id = random.choice(os.listdir(BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns'))
    im = np.zeros((630, 1890))
    for i, type in enumerate(['wbp', 'ctfdeconvolved', 'denoised']):
        zarr_path = BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns' / experiment_id / 'VoxelSpacing10.000' / f'{type}.zarr'
        arr = zarr.open(zarr_path)[0][64]
        arr = np.array(arr) * 1e4
        arr = arr / div + 0.5
        im[:, i*630:(i+1)*630] = arr
    im = im.clip(0, 1)
    im = (im * 255).astype(np.uint8)
    cv2.imwrite(f'./visualization_results/types/{experiment_id}.png', im)

def preprocess_playground():
    exp_id = 'TS_5_4'
    zarr_path = BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns' / exp_id / 'VoxelSpacing10.000' / 'wbp.zarr'
    slice = random.randint(0, 127)
    arr = zarr.open(zarr_path)[0][slice]
    arr = np.array(arr) * 2e4
    print(arr.std(), arr.mean())
    current = arr / 5 + 0.5

    #gaussian_kernel = cv2.getGaussianKernel(3, 1)
    #gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
    #arr = cv2.filter2D(arr, -1, gaussian_kernel)
    #sign = np.sign(arr)
    #arr = np.abs(arr)
    #arr = np.log1p(arr)
    #arr = arr ** 0.25
    #arr = arr * sign
    #print(arr.std(), arr.mean())
    low = np.percentile(arr, 0.1)
    high = np.percentile(arr, 99.9)
    arr = arr.clip(0)
    #arr = np.clip(arr, low, high)
    #arr = arr - low
    #arr = arr / (high - low)
    #arr = 1 - arr
    #arr = arr**0.25
    arr = arr / 2.5#5 + 0.5

    #arr = np.concatenate([current, arr], axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Show the original image
    axs[0, 0].imshow(current, cmap='gray', vmin=0, vmax=1)
    #axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Show the processed image
    axs[0, 1].imshow(arr, cmap='gray', vmin=0, vmax=1)
    #axs[0, 1].set_title('Processed Image')
    axs[0, 1].axis('off')

    # Show the histogram of the original image
    axs[1, 0].hist(current.flatten(), bins=100, color='gray')
    #axs[1, 0].set_title('Histogram of Original Image')

    # Show the histogram of the processed image
    axs[1, 1].hist(arr.flatten(), bins=100, color='gray')
    #axs[1, 1].set_title('Histogram of Processed Image')

    plt.tight_layout()
    plt.show()

import ndjson
def get_points_ext(experiment_id, jsonl_path):
    with open(
            jsonl_path
        ) as f:
        annotation_dicts = ndjson.load(f)
    points = []
    for d in annotation_dicts:
        points.append([d['location']['z'], d['location']['y'], d['location']['x']])
    return np.array(points)*10.012444 # multiply by 10 to convert to angstrom

def show_particle_counts():
    ext_ids = os.listdir(BASE_PATH / 'data' / 'external' / 'Simulated')
    df = {
        e: {
            'apo-ferritin': 0,
            'beta-galactosidase': 0,
            'ribosome': 0,
            'thyroglobulin': 0,
            'virus-like-particle': 0,
        } for e in ext_ids
    }
    for ext_id in ext_ids:
        for path, particle in zip([
            "Reconstructions/VoxelSpacing10.000/Annotations/101/ferritin_complex-1.0_orientedpoint.ndjson",
            "Reconstructions/VoxelSpacing10.000/Annotations/103/beta_galactosidase-1.0_orientedpoint.ndjson",
            "Reconstructions/VoxelSpacing10.000/Annotations/104/cytosolic_ribosome-1.0_orientedpoint.ndjson",
            "Reconstructions/VoxelSpacing10.000/Annotations/105/thyroglobulin-1.0_orientedpoint.ndjson",
            "Reconstructions/VoxelSpacing10.000/Annotations/106/pp7_vlp-1.0_orientedpoint.ndjson",
        ], ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']):
            path = BASE_PATH / 'data' / 'external' / 'Simulated' / ext_id / path
            points = get_points_ext(ext_id, path)
            df[ext_id][particle] = len(points)
    df = pd.DataFrame(df)
    print(df)
    df.to_csv('./visualization_results/particle_counts.csv')

def compare_ext_wbp(div=5):
    os.makedirs('./visualization_results/ext_vs_wbp', exist_ok=True)
    ext_ids = ['TS_0', 'TS_9', 'TS_18', 'TS_26']

    exp_id = random.choice(os.listdir(BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns'))
    wbp_path = BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns' / exp_id / 'VoxelSpacing10.000' / 'wbp.zarr'
    wbp = np.array(zarr.open(wbp_path)[0][64]) * 2e4# / div + 0.5
    std = wbp.std()
    print(std)

    im = np.zeros((1260, 2520))
    for i, ext_id in enumerate(ext_ids):
        zarr_path = BASE_PATH / 'data' / 'external' / 'Simulated' / ext_id / 'Reconstructions' / 'VoxelSpacing10.000' / 'Tomograms' / '100' / f'{ext_id}.zarr'
        arr = zarr.open(zarr_path)[0][64]
        arr = np.array(arr)
        high, low = np.percentile(arr, 97.725), np.percentile(arr, 2.275)
        sigma = (high - low) / 4
        if std**2 - sigma**2 > 0:
            # add noise so that the variance is 0.01
            print(sigma)
            noise_sigma = (std**2 - sigma**2)**0.5
            noise = np.random.randn(*arr.shape) #* noise_sigma
            # Convolve with a Gaussian kernel to introduce correlation
            kernel_size = 3
            kernel_sigma = 1
            kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
            kernel = kernel @ kernel.T
            correlated = cv2.filter2D(noise, -1, kernel)
            correlated = correlated * noise_sigma / correlated.std()
            arr = arr + correlated
            print(arr.std())

        #arr = arr / 8 + 0.5
        im[:630, i*630:(i+1)*630] = arr
        im[630:, i*630:(i+1)*630] = wbp
    im = im / 8 + 0.5
    im = im.clip(0, 1)
    im = (im * 255).astype(np.uint8)
    cv2.imwrite(f'./visualization_results/ext_vs_wbp/{exp_id}.png', im)

        

def plot_all_zarr_boxplots(type='wbp'):
    os.makedirs('./visualization_results', exist_ok=True)
    data_list = []
    labels = []
    zarr_dir = BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns'
    for experiment_ids in os.listdir(zarr_dir):
        fname = f'{experiment_ids}/VoxelSpacing10.000/{type}.zarr'
        full_path = os.path.join(zarr_dir, fname)
        arr = zarr.open(full_path)[0]
        arr = np.asarray(arr)
        print(np.percentile(arr, 99.9), np.percentile(arr, 0.1))
        data_list.append(arr.flatten())
        exp_id = fname.split('.')[0]  # or any custom parsing
        labels.append(f"({exp_id})")

    plt.boxplot(data_list, labels=labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./visualization_results/zarr_boxplots_{type}.png')
    plt.close()

def visualize_positive_ratio(train_sub_df):
    pos_cnt = {
        'apo-ferritin' : 47,
        'beta-amylase' : 5,
        'beta-galactosidase' : 6,
        'ribosome' : 80,
        'thyroglobulin' : 13,
        'virus-like-particle' : 6,
    }
    particle_cnt = {
        'apo-ferritin': 0,
        'beta-amylase': 0,
        'beta-galactosidase': 0,
        'ribosome': 0,
        'thyroglobulin': 0,
        'virus-like-particle': 0,
    }
    
    particle_radius = {
        'apo-ferritin': 60,  # apo-ferritin
        'beta-amylase': 65,  # beta-amylase
        'beta-galactosidase': 90,  # beta-galactosidase
        'ribosome': 150,  # ribosome
        'thyroglobulin': 130,  # thyroglobulin
        'virus-like-particle': 135,  # virus-like-particle
    }

    for idx, row in train_sub_df.iterrows():
        particle_cnt[row['particle_type']] += 1
    print(particle_cnt)
    pos_cnt = {k: v * particle_radius[k] ** 3 for k, v in pos_cnt.items()}

    # normalize by apo-ferritin
    pos_cnt = {k: v / pos_cnt['apo-ferritin'] for k, v in pos_cnt.items()}
    # take inverse for class weights
    pos_cnt = {k: 1/v for k, v in pos_cnt.items()}
    print(pos_cnt)

def visualize_ext_wbp_hist():
    os.makedirs('./visualization_results', exist_ok=True)
    for ext_exp in os.listdir(BASE_PATH / 'data' / 'external' / 'Simulated'):
        print(ext_exp)
        zarr_path = BASE_PATH / 'data' / 'external' / 'Simulated' / ext_exp / 'Reconstructions' / 'VoxelSpacing10.000' / 'Tomograms' / '100' / f'{ext_exp}.zarr'
        arr = zarr.open(zarr_path)[0]
        arr = np.array(arr).flatten()
        plt.hist(arr, bins=100)
        plt.savefig(f'./visualization_results/{ext_exp}.png')
        plt.clf()
    
    for exp in os.listdir(BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns'):
        print(exp)
        zarr_path = BASE_PATH / 'data' / 'original' / 'train' / 'static' / 'ExperimentRuns' / exp / 'VoxelSpacing10.000' / 'wbp.zarr'
        arr = zarr.open(zarr_path)[0]
        arr = np.array(arr).flatten()
        plt.hist(arr, bins=100)
        plt.savefig(f'./visualization_results/{exp}.png')
        plt.clf()

PARTICLE_RADIUS = {
    'apo-ferritin': 60,  # apo-ferritin
    'beta-amylase': 65,  # beta-amylase
    'beta-galactosidase': 90,  # beta-galactosidase
    'ribosome': 150,  # ribosome
    'thyroglobulin': 130,  # thyroglobulin
    'virus-like-particle': 135,  # virus-like-particle
}
classes=['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
from src.models.postprocessing import hmap_to_points_pooling, weighted_box_fusion
def visualize_preds(image, heatmaps, thresh=0.5, wbf_radius_multiplier=0.05):
    for i, c in enumerate(classes):
        pred_points = hmap_to_points_pooling(heatmaps[i], thresh)
        if wbf_radius_multiplier > 0:
            if not "confidence" in preds:
                print("Warning: confidence is not available for the weighted box fusion, setting all confidences to 1")
                preds["confidence"] = np.ones(len(preds["points"]))
            preds = weighted_box_fusion(preds, PARTICLE_RADIUS[c]*wbf_radius_multiplier)

COLORS = np.array([
    [],
])

particle_radius = {
    'apo-ferritin': 60,  # apo-ferritin
    'beta-amylase': 65,  # beta-amylase
    'beta-galactosidase': 90,  # beta-galactosidase
    'ribosome': 150,  # ribosome
    'thyroglobulin': 130,  # thyroglobulin
    'virus-like-particle': 135,  # virus-like-particle
}

from scipy.spatial import KDTree

def colorize_hmap(hmap, colors):
    colored = np.zeros((*hmap.shape[1:], 3))
    print(hmap.shape)
    for i, c in enumerate(colors):
        colored += hmap[i][..., None]*colors[i][None, None, None, :]
    return colored.astype('uint8')
    


classes=['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
def plot_predictions(path, image, heatmap, prediction, target, axis=0, step_size=8):
    """
    image: 3D numpy array
    heatmap: 3D numpy array with shape (n_classes, *image.shape)
    prediction: dictionary {class: np.array with shape (n_points, 3)}
    target: dictionary {class: np.array with shape (n_points, 3)}
    both the points should be in pixel coordinates, not angstrom
    """
    base_colors = np.array([
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ])

    tp_colors = base_colors*0.8
    tp_colors[4] = [66, 158, 233]
    fp_colors = (base_colors * 0.5 + 100).astype('uint8')
    fn_colors = (base_colors * 0.3).astype('uint8')
    color_samples = np.zeros((250, 150, 3))
    for i in range(3):
        clr = [tp_colors, fp_colors, fn_colors][i]
        for j in range(5):
            color_samples[j*50:(j+1)*50, i*50:(i+1)*50] = clr[j][None, None]
    cv2.imwrite(f'{path}/color_samples.png', color_samples)
            

    hmap = colorize_hmap(heatmap, base_colors)
    #print(hmap.shape)
    
    pred_results = {}
    target_results = {}
    for c in classes:
        pred_points = prediction[c]
        target_points = target[c]
        #print(pred_points, '\n', target_points)
        ref_tree = KDTree(target_points)
        pred_tree = KDTree(pred_points)
        raw_matches = pred_tree.query_ball_tree(ref_tree, r=particle_radius[c]/20)
        #print(raw_matches)
        pred_result = np.zeros(len(pred_points), dtype=bool)
        target_result = np.zeros(len(target_points), dtype=bool)
        for i, match in enumerate(raw_matches):
            if len(match) > 0:
                pred_result[i] = True
                target_result[match] = True
        pred_results[c] = pred_result
        target_results[c] = target_result
        #print(pred_result, target_result)

    for i in range(0, image.shape[axis], step_size):
        tomo_slice = image.take(i, axis=axis)
        #print(tomo_slice.shape)
        tomo_slice = tomo_slice[..., None].repeat(3, axis=-1)
        hmap_slice = hmap.take(i, axis=axis)
        #print(tomo_slice.shape, hmap_slice.shape)
        image_vis = np.concatenate([tomo_slice, hmap_slice], axis=1)
        for ci, c in enumerate(classes):
            pred_points = prediction[c]
            target_points = target[c]
            pred_result = pred_results[c]
            target_result = target_results[c]
            target_mask = (i - step_size/2 < target_points[:, axis]) & (target_points[:, axis] <= i + step_size/2)
            target_mask = target_mask | (i - particle_radius[c]/10 < target_points[:, axis]) & (target_points[:, axis] <= i + particle_radius[c]/10)
            pred_mask = (i - step_size/2 < pred_points[:, axis]) & (pred_points[:, axis] <= i + step_size/2)
            pred_mask = pred_mask | (i - particle_radius[c]/10 < pred_points[:, axis]) & (pred_points[:, axis] <= i + particle_radius[c]/10)

            #print(target_mask, target_result)
            #print(np.where(target_mask & target_result)[0])
            for tp in np.where(target_mask & target_result)[0]:
                tp_point = tuple(target_points[tp, 1:][[1, 0]].astype('int'))
                image_vis = cv2.circle(image_vis, tp_point, 2, tp_colors[ci].tolist(), -1)
                image_vis[tp_point[1], tp_point[0] + 630] = tp_colors[ci]
            
            for fp in np.where(pred_mask & ~pred_result)[0]:
                fp_point = tuple(pred_points[fp, 1:][[1, 0]].astype('int'))
                image_vis = cv2.circle(image_vis, fp_point, 2, fp_colors[ci].tolist(), -1)
                image_vis[fp_point[1], fp_point[0] + 630] = fp_colors[ci]
            
            for fn in np.where(target_mask & ~target_result)[0]:
                fn_point = tuple(target_points[fn, 1:][[1, 0]].astype('int'))
                image_vis = cv2.circle(image_vis, fn_point, 2, fn_colors[ci].tolist(), -1)
                image_vis[fn_point[1], fn_point[0] + 630] = fn_colors[ci]
        
        image_vis = cv2.resize(image_vis, fx=2.0, fy=2.0, dsize=None, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{path}/{i}.png', image_vis)



if __name__ == '__main__':
    train_sub_df = pd.read_csv(BASE_PATH / "data" / "processed" / "train_sub_df.csv")
    train_df = pd.read_csv(BASE_PATH / "data" / "processed" / "train_df.csv")
    arr = zarr.open(BASE_PATH / "data" / "original" / "train" / "static" / "ExperimentRuns" / "TS_5_4" / "VoxelSpacing10.000" / "denoised.zarr")[0]
    arr = np.asarray(arr)
    arr = arr * 1e5
    arr = arr.clip(
        np.percentile(arr, 1.0),
        np.percentile(arr, 99)
    )
    arr = arr[:64, 256:512, 256:512]
    #arr = arr.clip(0, 1)
    #arr = (arr * 255).astype(np.uint8)
#    visualize_intensity_distribution(BASE_PATH / "data" / "original" / "train" / "static" / "ExperimentRuns" / "TS_5_4" / "VoxelSpacing10.000" / "wbp.zarr")
    #visualize_intensity_distribution(BASE_PATH / "data" / "external" / "Simulated" / "TS_0" / "Reconstructions" / "VoxelSpacing10.000" / "Tomograms" / "100" / "TS_0.zarr")
    #visualize_positive_ratio(train_sub_df)
    #visualize_ext_wbp_hist()