import yaml
import json
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_configs(p_settings, p_cfg = None):
    with open(p_settings, 'r') as f:
        settings = json.load(f)
    if p_cfg is None:
        return settings
    with open(p_cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return settings, cfg
