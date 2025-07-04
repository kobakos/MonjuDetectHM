#!/usr/bin/env python3
"""
Example: Package and run inference with an ensemble of CryoET models using MLflow.

This script demonstrates how to:
1. Package multiple trained models as a single MLflow ensemble
2. Run ensemble inference with logit averaging
3. Compare single model vs ensemble performance
"""

import sys
import mlflow
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monjudetecthm.models.mlflow_integration import ModelPackager


def package_ensemble_model():
    """Package multiple models as an ensemble MLflow model."""
    base_config = 'configs/config_209.yml'
    base_weights_dir = 'results/209_resnet50d.ra2_in1k_20250116/weights'
    
    # Collect available fold weights
    config_paths = []
    weights_paths = []
    
    for fold in ['fold_0', 'fold_1', 'fold_2', 'fold_3']:
        weights_path = f'{base_weights_dir}/{fold}/best.pth'
        if Path(weights_path).exists():
            config_paths.append(base_config)
            weights_paths.append(weights_path)
    
    if len(weights_paths) < 2:
        return None
    
    # Create ensemble packager
    packager = ModelPackager(
        model_config_paths=config_paths,
        model_weights_paths=weights_paths,
        output_dir='mlflow_ensemble_model'
    )
    
    # Package the ensemble
    model_path = packager.package_model(model_name='cryoet_ensemble_model')
    return model_path


def run_ensemble_inference(model_path):
    """Run inference with the ensemble model."""
    copick_config_path = 'copick_config.json'
    if not Path(copick_config_path).exists():
        return False
    
    # Load the ensemble model
    ensemble_model = mlflow.pyfunc.load_model(model_path)
    
    # Load CoPick root
    import copick
    copick_root = copick.from_file(copick_config_path)
    
    if not copick_root.runs:
        return False
    
    experiment_id = copick_root.runs[0].name
    
    # Run ensemble inference
    model_input = {
        'copick_root': copick_root,
        'experiment_id': experiment_id
    }
    results = ensemble_model.predict(model_input)
    
    return True


def threshold_tuning_example(model_path):
    """Demonstrate threshold tuning for different sensitivity levels."""
    model = mlflow.pyfunc.load_model(model_path)
    import copick
    copick_root = copick.from_file('copick_config.json')
    
    if not copick_root.runs:
        return False
    
    experiment_id = copick_root.runs[0].name
    
    # Test different threshold levels
    threshold_levels = [
        [0.3, 0.3, 0.3, 0.3, 0.3],  # Low threshold (high sensitivity)
        [0.5, 0.5, 0.5, 0.5, 0.5],  # Default threshold
        [0.7, 0.7, 0.7, 0.7, 0.7],  # High threshold (low sensitivity)
    ]
    
    for threshold in threshold_levels:
        model_input = {
            'copick_root': copick_root,
            'experiment_id': experiment_id,
            'threshold': threshold
        }
        results = model.predict(model_input)
    
    return True


def main():
    # Step 1: Package the ensemble
    model_path = package_ensemble_model()
    if not model_path:
        return
    
    # Step 2: Run ensemble inference
    if not run_ensemble_inference(model_path):
        return
    
    # Step 3: Demonstrate threshold tuning
    if not threshold_tuning_example(model_path):
        return


if __name__ == "__main__":
    main()