"""
Model packaging utilities for creating MLflow models from trained CryoET models.
"""

import os
import shutil
import yaml
import subprocess
import tempfile
from typing import List, Union, Optional, Dict, Any
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import save_model as mlflow_save_model

from .mlflow_model import CryoETMLflowModel
from .utils import create_input_example


class ModelPackager:
    """
    Packages trained CryoET models as MLflow models with ensemble support.
    
    Supports both single models and ensembles with flexible dataset configuration.
    """
    
    def __init__(self,
                 model_config_path: Optional[Union[str, List[str]]] = None,
                 model_weights_path: Optional[Union[str, List[str]]] = None,
                 model_config_paths: Optional[List[str]] = None,
                 model_weights_paths: Optional[List[str]] = None,
                 output_dir: str = "mlflow_model"):
        """
        Initialize model packager.
        
        Args:
            model_config_path: Path to single model config (or list for ensemble)
            model_weights_path: Path to single model weights (or list for ensemble)  
            model_config_paths: List of paths to model configs (alternative to model_config_path)
            model_weights_paths: List of paths to model weights (alternative to model_weights_path)
            output_dir: Output directory for MLflow model
        """
        self.output_dir = output_dir
        
        # Handle different input formats
        if model_config_paths is not None:
            self.config_paths = model_config_paths
        elif model_config_path is not None:
            if isinstance(model_config_path, list):
                self.config_paths = model_config_path
            else:
                self.config_paths = [model_config_path]
        else:
            raise ValueError("Must provide model_config_path or model_config_paths")
            
        if model_weights_paths is not None:
            self.weights_paths = model_weights_paths
        elif model_weights_path is not None:
            if isinstance(model_weights_path, list):
                self.weights_paths = model_weights_path
            else:
                self.weights_paths = [model_weights_path]
        else:
            raise ValueError("Must provide model_weights_path or model_weights_paths")
            
        # Validate inputs
        if len(self.config_paths) != len(self.weights_paths):
            raise ValueError(f"Number of config files ({len(self.config_paths)}) must match number of weight files ({len(self.weights_paths)})")
            
        # Check if files exist
        for config_path in self.config_paths:
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
        for weights_path in self.weights_paths:
            if not Path(weights_path).exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
                
        self.ensemble_mode = len(self.config_paths) > 1
        
    def _create_requirements_txt(self) -> str:
        """
        Create requirements.txt with current environment dependencies.
        
        Returns:
            Path to created requirements.txt file
        """
        # Base requirements for CryoET model
        base_requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "timm>=0.9.0",
            "timm-3d>=1.0.0",
            "copick>=1.2.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "pyyaml>=6.0",
            "mlflow>=2.8.0",
            "opencv-python-headless>=4.5.0",
            "scipy>=1.7.0",
            "zarr>=2.10.0"
        ]
            
        # Write requirements file
        requirements_path = Path(self.output_dir) / "requirements.txt"
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(base_requirements))
            
        return str(requirements_path)
        
    def _prepare_artifacts(self) -> Dict[str, str]:
        """
        Prepare model artifacts for MLflow packaging.
        
        Returns:
            Dictionary mapping artifact names to file paths
        """
        artifacts = {}
        
        if self.ensemble_mode:
            # Ensemble: numbered artifacts
            for i, (config_path, weights_path) in enumerate(zip(self.config_paths, self.weights_paths)):
                artifacts[f'config_file_{i}'] = config_path
                artifacts[f'model_weights_{i}'] = weights_path
        else:
            # Single model
            artifacts['config_file'] = self.config_paths[0]
            artifacts['model_weights'] = self.weights_paths[0]
            
        return artifacts
        
    def package_model(self, 
                     model_name: Optional[str] = None,
                     mlflow_tracking: bool = False,
                     experiment_name: Optional[str] = None) -> str:
        """
        Package model(s) as MLflow model.
        
        Args:
            model_name: Name for the MLflow model
            mlflow_tracking: Whether to log to MLflow tracking server
            experiment_name: MLflow experiment name (if using tracking)
            
        Returns:
            Path to created MLflow model
        """
        if model_name is None:
            if self.ensemble_mode:
                model_name = f"cryoet_ensemble_{len(self.config_paths)}models"
            else:
                model_name = "cryoet_particle_detector"
                
        # Prepare artifacts
        artifacts = self._prepare_artifacts()
        
        # Create requirements.txt
        requirements_path = self._create_requirements_txt()
        
        # Create input example
        input_example = create_input_example()
        
        # Create MLflow model instance
        python_model = CryoETMLflowModel()
        
        # Package model
        model_path = str(Path(self.output_dir) / model_name)
        
        mlflow_save_model(
            path=model_path,
            python_model=python_model,
            artifacts=artifacts,
            pip_requirements=requirements_path,
            input_example=input_example,
            signature=None  # Will be inferred from input_example
        )
        
        # Log to MLflow tracking if requested
        if mlflow_tracking:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                
            with mlflow.start_run():
                # Log model artifacts
                mlflow.log_artifacts(model_path, "model")
                
                # Log model metadata
                mlflow.log_param("model_type", "ensemble" if self.ensemble_mode else "single")
                mlflow.log_param("num_models", len(self.config_paths))
                
                # Log model configs as artifacts
                for i, config_path in enumerate(self.config_paths):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    mlflow.log_dict(config, f"config_{i}.yml")
                    
        print(f"✅ Model {'ensemble' if self.ensemble_mode else ''} packaged successfully at: {model_path}")
        return model_path


def package_ensemble_from_experiments(experiment_dirs: List[str],
                                     folds: List[int], 
                                     weight_file: str = "best.pth",
                                     config_file: str = "config.yml",
                                     output_dir: str = "ensemble_model") -> str:
    """
    Package ensemble from multiple experiment directories.
    
    Args:
        experiment_dirs: List of experiment directory paths
        folds: List of fold numbers to use from each experiment
        weight_file: Name of weight file to use (e.g., "best.pth", "last.pth")
        config_file: Name of config file to use
        output_dir: Output directory for packaged model
        
    Returns:
        Path to packaged ensemble model
    """
    if len(experiment_dirs) != len(folds):
        raise ValueError("Number of experiment directories must match number of folds")
        
    config_paths = []
    weights_paths = []
    
    for exp_dir, fold in zip(experiment_dirs, folds):
        exp_path = Path(exp_dir)
        
        # Look for config file
        config_path = exp_path / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config_paths.append(str(config_path))
        
        # Look for weight file
        weights_path = exp_path / "weights" / f"fold_{fold}" / weight_file
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        weights_paths.append(str(weights_path))
        
    # Create packager and package ensemble
    packager = ModelPackager(
        model_config_paths=config_paths,
        model_weights_paths=weights_paths,
        output_dir=output_dir
    )
    
    return packager.package_model()