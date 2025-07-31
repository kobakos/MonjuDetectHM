
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

import copick


class ModelPackager:
    """
    Packages trained CryoET models as MLflow models with ensemble support.
    
    Supports both single models and ensembles with flexible dataset configuration.
    """
    
    def __init__(self,
                 model_dir: Union[str, List],
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
        self.model_dir = model_dir if isinstance(model_dir, list) else [model_dir]
        self.output_dir = output_dir
                
        self.ensemble_mode = len(self.model_dir) > 1
        
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
        for i, model_dir in enumerate(self.model_dir):
            artifacts[f'model_dir_{i}'] = model_dir
        return artifacts
        
    def package_model(self, 
                     model_name: Optional[str] = None):
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
            model_name = "cryoet_particle_detector"
                
        # Prepare artifacts
        artifacts = self._prepare_artifacts()
        
        # Create requirements.txt
        requirements_path = self._create_requirements_txt()
        
        # Create MLflow model instance
        python_model = CryoETMLflowModel()
        
        # Package model
        model_path = str(Path(self.output_dir) / model_name)
        
        mlflow_save_model(
            path=model_path,
            python_model=python_model,
            artifacts=artifacts,
            pip_requirements=requirements_path,
            signature=None,
        )
                    
        print(f"Model {'ensemble' if self.ensemble_mode else ''} packaged successfully at: {model_path}")
        return model_path
