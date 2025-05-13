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
czii_model_contribution/
├── configs/          # Configuration files for training and inference
├── data/             # Data directory (can be restructured as needed)
├── src/              # Source code
│   ├── component_factory.py   # Factory for creating optimizers, schedulers, criteria, and metrics
│   ├── constants.py           # Defines constants (e.g., particle classes and radii)
│   ├── data_processing/     # Data loading and preprocessing
│   │   ├── augmentations.py # Data augmentation techniques
│   │   ├── dataset.py       # PyTorch Dataset classes
│   │   ├── input_preprocessor.py # Input preprocessing steps
│   │   ├── target_generator.py   # Target generation for training
│   │   └── init.py
│   ├── evaluation.py        # Evaluation metrics
│   ├── logger.py            # Logging and model saving
│   ├── losses.py            # Custom loss functions
│   ├── models/              # Model architecture
│   │   ├── init.py
│   │   ├── decoder.py       # Decoder modules
│   │   ├── models.py        # Main model definition
│   │   ├── postprocessing.py  # Post-processing of model outputs
│   │   ├── prepare_3d_model.py # Model setup and conversion
│   ├── utils/             # Utility functions
│   │   ├── init.py
│   │   ├── loop.py          # Training loop utilities
│   │   ├── mpl_ssh.py       # Matplotlib SSH configuration
│   │   └── utils.py         # General utility functions
│   └── init.py
├── create_df.py      # Script to create training and validation DataFrames
├── download_external.py # Script to download external data
├── SETTINGS.json       # Project settings (paths, etc.)
├── train.py          # Training script
├── validate.py       # Validation/inference script
├── visualization.py  # Visualization tools
└── README.md         # This file
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd czii_model_contribution
    ```

2.  **Set up the environment**

    ```bash
    pip install uv # if you don't have it already
    ```
    * Create and sync virtual environment using `uv`:

    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

3.  **Configure `SETTINGS.json`:**

    * Update the paths in `SETTINGS.json` to point to your data directories.

## Data Preparation

1.  **Download External Data (Optional):**

    * If using external simulated data, run:

        ```bash
        python download_external.py
        ```

2.  **Create DataFrames:**

    * Run the `create_df.py` script to generate the necessary training and validation DataFrames:

        ```bash
        python create_df.py
        ```

## Training

1.  **Configure Training:**

    * Modify the YAML files in the `configs/` directory to adjust training parameters, model architecture, and data augmentations.
    * Example configurations are provided (e.g., `config_209.yml`, `config_249.yml`, `config_265.yml`).

2.  **Start Training:**

    * Run the `train.py` script, specifying the configuration file:

        ```bash
        python train.py --config configs/config_209.yml --log --tqdm
        ```

    * Optional arguments:
        * `--log`: Enable logging (e.g., to Weights & Biases).
        * `--tqdm`:  Use tqdm progress bars.
        * `--detect_anomaly`: Enable PyTorch anomaly detection.
        * `--sanity_check`:  Run a quick sanity check.

## Validation/Inference

1.  **Configure Inference:**

    * Use the `validate.py` script with an appropriate configuration (e.g., `configs/infer_config.yml`).

2.  **Run Inference:**

    ```bash
    python validate.py --config configs/infer_config.yml --tqdm --show
    ```

    * Optional arguments:
        * `--tqdm`:  Use tqdm progress bars.
        * `--show`:  Save visualization results.
        * `--more-tolerant`: Use a less strict evaluation metric.

## Model Description

The core of the project is a 3D convolutional neural network model (`src/models/models.py`) built using the `timm` library. Key aspects include:

* **Backbone:** Configurable (e.g., ResNet, EfficientNet)
* **Decoder:** Unet or DeepLabV3+
* **Input/Output:** 3D Cryo-ET volumes and corresponding particle density maps.
* **Loss Function:** Configurable (e.g., BCE, Focal Loss)
* **Optimizer/Scheduler:** Configurable (e.g., AdamW, Cosine Annealing)

## Data Handling

* **DataFrames:** `create_df.py` generates pandas DataFrames that manage training and validation data, including paths to data, crop origins, and folds for cross-validation.
* **Dataset Class:** The `src/data_processing/dataset.py`  module defines PyTorch Dataset classes to efficiently load and preprocess Cryo-ET data.
* **Augmentations:** `src/data_processing/augmentations.py`  provides a set of 3D data augmentation techniques to improve model robustness.

## Evaluation

* The `src/evaluation.py` module contains the scoring metric used to evaluate the model's performance, considering true positives, false positives, and false negatives.
## License

MIT

## Acknowledgements

* Kaggle for providing the dataset and competition.
* The developers of the libraries used in this project (PyTorch, timm, etc.).