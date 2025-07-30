# Cryo-ET Particle Detection Model

## Overview

This project contains the code for a deep learning model designed for particle detection in Cryo-Electron Tomography (Cryo-ET) data. The model was developed for a Kaggle competition (https://www.kaggle.com/competitions/czii-cryo-et-object-identification). It handles data preprocessing, model training, validation, and inference, with a focus on accuracy.

## Key Features

* **3D Deep Learning Model:** Utilizes a 3D convolutional neural network architecture for precise particle detection in Cryo-ET volumes.
* **Configurable Training:** Training parameters, model architectures, and data augmentations are managed through YAML configuration files, allowing for easy experimentation.
* **Data Augmentation:** Includes a suite of 3D augmentations to improve model robustness and generalization.
* **External Data Handling:** Supports the integration of external (simulated) data to enhance training.

## Project Structure

The project is organized as follows:

```
MonjuDetectHM/
├── configs/          # Configuration files for training and inference
├── monjudetecthm/      # Source code
│   ├── component_factory.py   # Factory for creating optimizers, schedulers, criteria, and metrics
│   ├── data_processing/     # Data loading and preprocessing
│   │   ├── augmentations.py # Data augmentation techniques
│   │   ├── dataset.py       # PyTorch Dataset classes
│   │   ├── input_preprocessor.py # Input preprocessing steps
│   │   └── target_generator.py   # Target generation for training
│   ├── evaluation.py        # Evaluation metrics
│   ├── losses.py            # Custom loss functions
│   ├── models/              # Model architecture
│   │   ├── decoder_deeplab.py # DeepLabV3+ decoder
│   │   ├── decoder_unet.py    # UNet decoder
│   │   ├── models.py        # Main model definition
│   │   ├── postprocessing.py  # Post-processing of model outputs
│   │   ├── prepare_3d_model.py # Model setup and conversion
│   │   └── mlflow_integration/ # MLFlow integration for model packaging
│   └── utils/             # Utility functions
│       ├── loop.py          # Training loop utilities
│       ├── mpl_ssh.py       # Matplotlib SSH configuration
│       └── utils.py         # General utility functions
├── example_ensemble_model.py # Example for packaging and running an ensemble of models
├── example_single_model.py   # Example for packaging and running a single model
├── run_inference.py  # Script for running inference
├── train.py          # Training script
├── pyproject.toml    # Project metadata and dependencies
├── README.md         # This file
└── ...
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd MonjuDetectHM
    ```

2.  **Set up the environment**

    * Create and sync virtual environment using `uv`:

    ```bash
    pip install uv # if you don't have it already
    uv venv
    source .venv/bin/activate
    uv sync
    ```

## Model Description

The core of the project is a 3D convolutional neural network model (`monjudetecthm/models/models.py`) built using the `timm` library. Key aspects include:

* **Backbone:** Configurable (e.g., ResNet, EfficientNet)
* **Decoder:** Unet or DeepLabV3+
* **Input/Output:** 3D Cryo-ET volumes and corresponding particle density maps.
* **Loss Function:** Configurable (e.g., BCE, Focal Loss)
* **Optimizer/Scheduler:** Configurable (e.g., AdamW, Cosine Annealing)

## Data Handling

* **Dataset Class:** The `monjudetecthm/data_processing/dataset.py`  module defines PyTorch Dataset classes to efficiently load and preprocess Cryo-ET data.
* **Augmentations:** `monjudetecthm/data_processing/augmentations.py`  provides a set of 3D data augmentation techniques to improve model robustness.

## Evaluation

* The `monjudetecthm/evaluation.py` module contains the scoring metric used to evaluate the model's performance, considering true positives, false positives, and false negatives.

## Running Sample Scripts

### Data Setup

Before running any scripts, you need to download the competition data:

1. **Download the data using Kaggle API:**
   ```bash
   kaggle competitions download -c czii-cryo-et-object-identification
   ```

2. **Extract and organize the data:**
   ```bash
   unzip czii-cryo-et-object-identification.zip -d data/original/
   ```
   
   The data should be structured as follows:
   ```
   data/
   ├── original/
   │   ├── train/
   │   │   ├── overlay/
   │   │   └── static/
   ...
   ```

### Download Pretrained Models

Download the pretrained models from 

https://drive.google.com/drive/folders/1hu1K1hGAkn-l-Mmn53P_XumvTtVEdrTj?usp=sharing

and place the `models` folder under the project root.
It should look like:

```
models
├── 209_resnet50d.ra2_in1k_20250116
│   ├── config.yml
│   └── weights
│       ├── fold_0
│       │   └── last.pth
│       ├── fold_1
│       │   └── best.pth
|       ...
├── 249_tf_efficientnetv2_m.in21k_ft_in1k_20250125
...
```

### Sample Scripts

#### 1. Training Script (`train.py`)

Train a model from scratch using a configuration file:

```bash
python train.py --config configs/config_209.yml --copick_config_path copick_config.json
```

Optional parameters:
- `--tqdm`: Show progress bars
- `--sanity_check`: Run with reduced steps for testing
- `--model_save_dir results`: Directory to save trained models (default: `results`)

#### Configure the experiment

* Modify the YAML files in the `configs/` directory to adjust training parameters, model architecture, data augmentations etc.
* Example configurations are provided (e.g., `config_209.yml`, `config_249.yml`, `config_265.yml`).

#### 2. Inference Script (`run_inference.py`)

Run inference using a pretrained model:

```bash
python run_inference.py --config configs/infer_config.yml --copick_config copick_config.json --model_path models/209_resnet50d.ra2_in1k_20250116
```

Optional parameters:
- `--tqdm`: Show progress bars
- `--detect_anomaly`: Enable anomaly detection

#### 3. Single Model Example (`example_single_model.py`)

Package and run inference with a single model using MLflow:

```bash
python example_single_model.py
```

This script demonstrates:
- Packaging a trained PyTorch model as an MLflow model
- Loading the MLflow model for inference
- Running particle detection on 3D tomography data

#### 4. Ensemble Model Example (`example_ensemble_model.py`)

Package and run inference with an ensemble of models:

```bash
python example_ensemble_model.py
```

This script demonstrates:
- Packaging multiple trained models as a single MLflow ensemble
- Running ensemble inference with logit averaging
- Threshold tuning for different sensitivity levels

### Notes

- **CoPick Configuration**: The `copick_config.json` file is already configured to work with the expected data structure.
- **GPU Usage**: All scripts are configured to use CUDA by default. Ensure you have a compatible GPU and CUDA installation.
