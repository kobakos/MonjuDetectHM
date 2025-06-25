"""
Utility functions for MLflow integration with CryoET models.
"""

import yaml
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from monjudetecthm.models.models import build_model_from_config


def load_model_ensemble(config_paths: List[str], 
                       weights_paths: List[str],
                       device: Optional[torch.device] = None) -> List[torch.nn.Module]:
    """
    Load an ensemble of CryoET models from config and weight files.
    
    Args:
        config_paths: List of paths to model configuration files
        weights_paths: List of paths to model weight files
        device: Device to load models on (default: auto-detect)
        
    Returns:
        List of loaded PyTorch models
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if len(config_paths) != len(weights_paths):
        raise ValueError(f"Number of config files ({len(config_paths)}) must match number of weight files ({len(weights_paths)})")
        
    models = []
    
    for config_path, weights_path in zip(config_paths, weights_paths):
        # Load configuration
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        model_cfg = cfg['model'].copy()
        
        # Remove EMA and pretrained settings for inference
        if 'ema' in model_cfg:
            del model_cfg['ema']
        model_cfg['pretrained'] = False
        if 'pretrain_weight_path' in model_cfg:
            del model_cfg['pretrain_weight_path']
            
        # Set device
        model_cfg['device'] = device
        
        # Build model
        model = build_model_from_config(model_cfg)
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=device)
        
        # Handle compiled model weights
        if '_orig_mod.segmentation_head.weight' in state_dict:
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.eval()
        
        models.append(model)
        
    return models


def ensemble_inference(models: List[torch.nn.Module], 
                      input_tensor: torch.Tensor,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Run ensemble inference with logit averaging.
    
    Args:
        models: List of PyTorch models
        input_tensor: Input tensor for inference
        device: Device for computation (default: auto-detect)
        
    Returns:
        Averaged model output tensor
    """
    if device is None:
        device = input_tensor.device
        
    if len(models) == 1:
        # Single model - no averaging needed
        with torch.no_grad():
            return models[0](input_tensor)
            
    # Ensemble inference with logit averaging
    ensemble_output = None
    
    with torch.no_grad():
        for model in models:
            output = model(input_tensor.to(device))
            
            # Convert to float for averaging
            output = output.float()
            
            if ensemble_output is None:
                ensemble_output = output
            else:
                ensemble_output += output
                
    # Average the outputs
    ensemble_output /= len(models)
    
    return ensemble_output


def create_input_example(image_size: Tuple[int, int, int] = (128, 128, 128)) -> pd.DataFrame:
    """
    Create example input data for MLflow model signature inference.
    
    Args:
        image_size: Size of 3D image volume
        
    Returns:
        DataFrame with example input data
    """
    # Create dummy 3D image data
    dummy_image = np.random.randn(*image_size).astype(np.float32)
    
    # Create example DataFrame with CoPick config path included
    input_example = pd.DataFrame({
        'image_data': [dummy_image],
        'crop_origin': [np.array([0, 0, 0])],
        'experiment_id': ['example_experiment'],
        'copick_config_path': ['copick_config.json']  # Include CoPick config path in input
    })
    
    return input_example


def validate_model_compatibility(config_paths: List[str]) -> bool:
    """
    Validate that models in ensemble are compatible for averaging.
    
    Args:
        config_paths: List of paths to model configuration files
        
    Returns:
        True if models are compatible, False otherwise
    """
    if len(config_paths) <= 1:
        return True
        
    # Load all configs
    configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        configs.append(cfg)
        
    # Check critical compatibility requirements
    base_config = configs[0]
    
    for cfg in configs[1:]:
        # Check image size compatibility
        if cfg['dataset']['image_size'] != base_config['dataset']['image_size']:
            print(f"Warning: Image size mismatch - {cfg['dataset']['image_size']} vs {base_config['dataset']['image_size']}")
            return False
            
        # Check number of classes
        if cfg['model']['n_classes'] != base_config['model']['n_classes']:
            print(f"Warning: Number of classes mismatch - {cfg['model']['n_classes']} vs {base_config['model']['n_classes']}")
            return False
            
        # Check voxel spacing
        if cfg['dataset']['voxel_spacing'] != base_config['dataset']['voxel_spacing']:
            print(f"Warning: Voxel spacing mismatch - {cfg['dataset']['voxel_spacing']} vs {base_config['dataset']['voxel_spacing']}")
            
    return True


def get_model_info(config_path: str, weights_path: str) -> Dict[str, Any]:
    """
    Extract model information from config and weights files.
    
    Args:
        config_path: Path to model configuration file
        weights_path: Path to model weights file
        
    Returns:
        Dictionary with model information
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Get model info
    model_cfg = cfg.get('model', {})
    dataset_cfg = cfg.get('dataset', {})
    
    # Get weights info
    weights_path = Path(weights_path)
    weights_size = weights_path.stat().st_size if weights_path.exists() else 0
    
    info = {
        'model_name': model_cfg.get('name', 'unknown'),
        'architecture': model_cfg.get('architecture', 'unknown'),
        'n_classes': model_cfg.get('n_classes', 0),
        'image_size': dataset_cfg.get('image_size', [0, 0, 0]),
        'voxel_spacing': dataset_cfg.get('voxel_spacing', 0.0),
        'weights_file': str(weights_path),
        'weights_size_mb': round(weights_size / (1024 * 1024), 2),
        'config_file': config_path
    }
    
    return info


def prepare_inference_data(image_data: np.ndarray,
                          crop_origin: np.ndarray,
                          experiment_id: str) -> pd.DataFrame:
    """
    Prepare data for MLflow model inference.
    
    Args:
        image_data: 3D numpy array of tomography data
        crop_origin: 3D coordinates [d0, d1, d2]
        experiment_id: experiment identifier
        
    Returns:
        DataFrame formatted for MLflow model input
    """
    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D image data, got {image_data.ndim}D")
        
    if len(crop_origin) != 3:
        raise ValueError(f"Expected 3D crop origin, got {len(crop_origin)}D")
        
    input_df = pd.DataFrame({
        'image_data': [image_data.astype(np.float32)],
        'crop_origin': [np.array(crop_origin)],
        'experiment_id': [experiment_id]
    })
    
    return input_df


def format_prediction_results(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format prediction results into a structured DataFrame.
    
    Args:
        predictions: List of prediction dictionaries from MLflow model
        
    Returns:
        DataFrame with formatted prediction results
    """
    rows = []
    
    for i, pred_dict in enumerate(predictions):
        for experiment_id, experiment_preds in pred_dict.items():
            for particle_class, detections in experiment_preds.items():
                if 'points' in detections and len(detections['points']) > 0:
                    points = detections['points']
                    scores = detections.get('scores', [1.0] * len(points))
                    
                    for point, score in zip(points, scores):
                        rows.append({
                            'prediction_id': i,
                            'experiment_id': experiment_id,
                            'particle_class': particle_class,
                            'x': point[0],
                            'y': point[1], 
                            'z': point[2],
                            'confidence': score
                        })
                        
    if not rows:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'prediction_id', 'experiment_id', 'particle_class',
            'x', 'y', 'z', 'confidence'
        ])
        
    return pd.DataFrame(rows)