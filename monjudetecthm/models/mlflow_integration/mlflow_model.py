"""
MLflow wrapper model for CryoET particle detection with ensemble support.
"""

import os
import yaml
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel
import copick

from monjudetecthm.models.models import build_model_from_config
from monjudetecthm.models.postprocessing import PostProcessor
from monjudetecthm.evaluation import pred_dicts_to_df
from monjudetecthm.data_processing.dataset import CropDataset, generate_sliding_window_index


class CryoETMLflowModel(PythonModel):
    """
    MLflow wrapper for CryoET particle detection models with ensemble support.
    
    This class provides a standardized interface for loading and running inference
    with single or ensemble CryoET models using CoPick root objects. It handles
    sliding window generation, preprocessing, and aggregation of results automatically.
    """
    
    def __init__(self):
        self.models = []
        self.model_configs = []
        self.device = None
        self.ensemble_mode = False
        
    def load_context(self, context):
        """
        Load model(s) and configuration from MLflow context.
        
        Args:
            context: MLflow context containing model artifacts
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configurations
        config_files = []
        if 'config_file' in context.artifacts:
            # Single model
            config_files = [context.artifacts['config_file']]
        else:
            # Ensemble - look for numbered config files
            config_files = []
            i = 0
            while f'config_file_{i}' in context.artifacts:
                config_files.append(context.artifacts[f'config_file_{i}'])
                i += 1
        
        if not config_files:
            raise ValueError("No configuration files found in MLflow artifacts")
            
        # Load model weights
        weights_files = []
        if 'model_weights' in context.artifacts:
            # Single model
            weights_files = [context.artifacts['model_weights']]
        else:
            # Ensemble - look for numbered weight files
            weights_files = []
            i = 0
            while f'model_weights_{i}' in context.artifacts:
                weights_files.append(context.artifacts[f'model_weights_{i}'])
                i += 1
                
        if not weights_files:
            raise ValueError("No model weight files found in MLflow artifacts")
            
        if len(config_files) != len(weights_files):
            raise ValueError(f"Mismatch between config files ({len(config_files)}) and weight files ({len(weights_files)})")
            
        # Set ensemble mode
        self.ensemble_mode = len(config_files) > 1
        
        # Load models
        self.models = []
        self.model_configs = []
        
        for config_file, weights_file in zip(config_files, weights_files):
            # Load configuration
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
            
            model_cfg = cfg['model'].copy()
            
            # Remove EMA and pretrained settings for inference
            if 'ema' in model_cfg:
                del model_cfg['ema']
            model_cfg['pretrained'] = False
            if 'pretrain_weight_path' in model_cfg:
                del model_cfg['pretrain_weight_path']
            
            # Set device
            model_cfg['device'] = self.device
            
            # Build model
            model = build_model_from_config(model_cfg)
            
            # Load weights
            state_dict = torch.load(weights_file, map_location=self.device)
            
            # Handle compiled model weights
            
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            model.eval()
            
            self.models.append(model)
            self.model_configs.append(cfg)
            
        print(f"Loaded {'ensemble of ' + str(len(self.models)) + ' models' if self.ensemble_mode else 'single model'} on {self.device}")
        
    def predict(self, context, model_input) -> Dict:
        """
        Run inference on CoPick root data.
        
        Args:
            context: MLflow context (unused)
            model_input: Dictionary containing:
                - 'copick_root': CoPick root object containing tomogram data
                - 'experiment_id': Optional experiment ID to process (if None, processes all experiments)
                - 'threshold': Optional detection thresholds per class
                - 'voxel_spacing': Optional voxel spacing override (if None, uses config value)
                
        Returns:
            Dictionary with particle detections aggregated across all processed crops
        """
        if model_input is None:
            raise ValueError("model_input must be provided")
        
        if not isinstance(model_input, dict):
            raise ValueError("model_input must be a dictionary with 'copick_root' key")
        
        # Extract parameters from model_input
        copick_root = model_input.get('copick_root')
        experiment_id = model_input.get('experiment_id')
        voxel_spacing = model_input.get('voxel_spacing')
        threshold = model_input.get('threshold')

        
        if copick_root is None:
            raise ValueError("'copick_root' must be provided in model_input")
        
        # Use the first model's config for dataset and postprocessing settings
        base_cfg = self.model_configs[0]
        voxel_spacing = base_cfg['dataset']['voxel_spacing'] if voxel_spacing is None else voxel_spacing
        
        # Generate sliding window index for all experiments
        df = generate_sliding_window_index(
            copick_root=copick_root,
            voxel_spacing=voxel_spacing,
            image_size=base_cfg['dataset']['image_size'],
            image_stride=base_cfg['dataset']['image_stride'],
            include_edge_windows=True
        )
        
        # Filter by experiment_id if provided
        if experiment_id is not None:
            df = df[df['experiment_id'] == experiment_id]
            if df.empty:
                raise ValueError(f"No data found for experiment_id: {experiment_id}")
        
        # Create dataset for inference
        dataset = CropDataset(
            copick_root=copick_root,
            df=df,
            index=np.arange(len(df)),  # Use all indices
            voxel_spacing=base_cfg['dataset']['voxel_spacing'],
            image_size=base_cfg['dataset']['image_size'],
            return_targets=False,  # Inference only
            do_augmentation=False,  # No augmentation for inference
            rescale_factor=base_cfg['dataset'].get('rescale_factor', 1e5),
            clip_percentile=base_cfg['dataset'].get('clip_percentile', (0.1, 99.9)),
            standardize=base_cfg['dataset'].get('standardize', False)
        )
        
        # Create dataloader with batch size 1
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single threaded for inference
            collate_fn=dataset.collate_fn
        )
        
        # Initialize postprocessor
        postprocessing_cfg = base_cfg.get("postprocessing", {}).copy()
        post_processor = PostProcessor(
            copick_root=copick_root,
            window_size=base_cfg['dataset']['image_size'],
            window_stride=base_cfg['dataset']['image_stride'], 
            voxel_spacing=base_cfg['dataset']['voxel_spacing'],
            selected_classes=base_cfg['dataset']['classes'],
            **postprocessing_cfg
        )
        
        # Override thresholds if provided
        if threshold is not None:
            post_processor.threshold = threshold
        
        # Process all crops through models
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            
            # Get corresponding crop information from DataFrame
            df_row = df.iloc[batch_idx]
            crop_origin = np.array([df_row['crop_origin_d0'], df_row['crop_origin_d1'], df_row['crop_origin_d2']])
            exp_id = df_row['experiment_id']
            
            # Run ensemble inference
            with torch.no_grad():
                if self.ensemble_mode:
                    # Ensemble: average logits before activation
                    ensemble_output = 0
                    for model in self.models:
                        output = model(images)
                        ensemble_output += output.float()
                    ensemble_output /= len(self.models)
                    final_output = ensemble_output
                else:
                    # Single model
                    final_output = self.models[0](images)
            
            # Accumulate results in postprocessor
            post_processor.accumulate(
                final_output,
                crop_origins=crop_origin.reshape(1, -1),
                experiment_ids=[exp_id]
            )
        
        # Get final predictions
        return post_processor.predictions