#!/usr/bin/env python3
"""
Example: Package and run inference with a single CryoET model using MLflow.

This script demonstrates how to:
1. Package a trained PyTorch model as an MLflow model
2. Load the MLflow model for inference
3. Run particle detection on 3D tomography data
"""

import sys
import mlflow
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from monjudetecthm.models.mlflow_integration import ModelPackager

def package_single_model():
    """Package a single trained model as MLflow model."""
    model_dir = "models/209_resnet50d.ra2_in1k_20250116"
    output_dir = 'mlflow_single_model'
    
    # Create packager
    packager = ModelPackager(
        model_dir=model_dir,
        output_dir=output_dir
    )
    
    # Package the model
    model_path = packager.package_model(model_name='cryoet_single_model')
    return model_path

def run_inference_example(model_path):
    """Run inference example with the packaged model."""
    copick_config_path = 'copick_config.json'
    if not Path(copick_config_path).exists():
        return False
    
    # Load the MLflow model
    model = mlflow.pyfunc.load_model(model_path)
    
    # Load CoPick root
    import copick
    copick_root = copick.from_file(copick_config_path)
    
    if not copick_root.runs:
        return False
    
    experiment_id = copick_root.runs[0].name
    
    # Run inference
    model_input = {
        'copick_root': copick_root,
        'experiment_id': experiment_id
    }
    results = model.predict(model_input)
    
    return results


def multi_experiment_example(model_path):
    """Demonstrate processing multiple experiments."""
    model = mlflow.pyfunc.load_model(model_path)
    import copick
    copick_root = copick.from_file('copick_config.json')
    
    # Process all experiments at once (experiment_id=None)
    model_input = {
        'copick_root': copick_root,
        'experiment_id': None
    }
    results = model.predict(model_input)
    
    return results


def main():
    """Main example workflow."""
    # Step 1: Package the model
    model_path = package_single_model()
    print("MlFlow model exported to:", model_path)
    model_path = 'mlflow_single_model/cryoet_single_model'  # Use pre-packaged model for demo
    results = run_inference_example(model_path)
    print("Inference results:", results)


if __name__ == "__main__":
    main()