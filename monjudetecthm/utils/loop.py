import torch


import tqdm

def to_device_recursive(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device_recursive(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: to_device_recursive(v, device) for k, v in data.items()}
    elif torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, (int, float)):
        return data
    else:
        print(f"Warning: to_device_recursive: {type(data)} not handled")
        return data
    
def to_device(images, labels, device):
    images = images.to(device, non_blocking=True)
    if isinstance(labels, (list, tuple)):
        labels = to_device_recursive(labels, device)
    elif isinstance(labels, dict):
        labels = {k: to_device_recursive(v, device) for k, v in labels.items()}
    else:
        labels = labels.to(device, non_blocking=True)
    return images, labels