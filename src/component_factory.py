import os
import cv2
import tqdm
import torch
from torch import nn
import torch.nn.functional as F

import torch

# code is from https://github.com/davda54/sam
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def create_optimizer(cfg, parameters):
    optimizer = cfg['name']
    if optimizer == 'adam':
        return torch.optim.Adam(parameters, **cfg.get('args'))
    elif optimizer == 'adamw':
        return torch.optim.AdamW(parameters, **cfg.get('args'))
    elif optimizer == 'sgd':
        return torch.optim.SGD(parameters, **cfg.get('args'))
    elif optimizer == 'adan':
        from adan_pytorch import Adan
        return Adan(parameters, **cfg.get('args'))
    elif optimizer == 'sam':
        return SAM(parameters, torch.optim.AdamW, **cfg.get('args'))
    else:
        raise ValueError(f'Unknown optimizer: {optimizer}')
    
def list_available_optimizers():
    return ['adam', 'adamw', 'sgd', 'adan']
    
def create_scheduler(cfg, optimizer,):
    scheduler = cfg['name']
    kwargs = cfg.get('args')
    if scheduler is None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif scheduler == 'cosine_timm':
        
        from timm.scheduler import CosineLRScheduler
        return CosineLRScheduler(optimizer, **kwargs)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler == 'step':
        pass
    else:
        raise ValueError(f'Unknown scheduler: {scheduler}')

def list_available_schedulers():
    return ['cosine_timm', 'cosine', 'plateau']

class MicroAvgDiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, y, t):
        y = F.softmax(y, dim = 1)
        y = y.flatten(start_dim = 1)
        y = y[:, 1:]
        t = t.flatten(start_dim = 1)
        t = t[:, 1:]
        intersection = (y * t).sum(axis=-1)
        union = y.sum() + t.sum(axis=-1)
        dice = (2. * intersection) / (union + self.eps)
        return 1 - dice.mean()
    
class DiceLoss_binary(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, y, t):
        y = F.sigmoid(y)
        y = y.flatten(start_dim = 1)
        t = t.flatten(start_dim = 1)
        intersection = (y * t).sum(axis=-1)
        union = y.sum(axis=-1) + t.sum(axis=-1)
        dice = (2. * intersection) / (union + self.eps)
        return 1 - dice.mean()
    
class MacroAvgDiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, y, t):
        y = F.softmax(y, dim = 1)
        y = y.flatten(start_dim = 2)
        t = t.flatten(start_dim = 2)
        intersection = (y * t).sum(axis=-1)
        union = y.sum(axis=-1) + t.sum(axis=-1)
        dice = (2. * intersection) / (union + self.eps)
        return 1 - dice.mean()
    
class CE_DiceLoss(torch.nn.Module):
    def __init__(self, ce_weight = 0.5, averaging = 'micro', eps=1e-7):
        super().__init__()
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction = 'mean')
        self.dice = MicroAvgDiceLoss(eps) if averaging == "micro" else MacroAvgDiceLoss(eps)
        self.ce_weight = ce_weight
    def forward(self, y, t):
        ce = self.ce(y.flatten(start_dim = 2), t.flatten(start_dim = 2))
        dice = self.dice(y, t)
        return self.ce_weight * ce + (1 - self.ce_weight) * dice
    
class BCE_DiceLoss(torch.nn.Module):
    def __init__(self, ce_weight = 0.5, averaging = 'micro', eps=1e-7):
        super().__init__()
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss()
        #self.dice = MicroAvgDiceLoss(eps) if averaging == "micro" else MacroAvgDiceLoss(eps)
        self.dice = DiceLoss_binary(eps)
        self.ce_weight = ce_weight
    def forward(self, y, t):
        ce = self.bce(y.flatten(start_dim = 1), t.flatten(start_dim = 1))
        dice = self.dice(y, t)
        return self.ce_weight * ce + (1 - self.ce_weight) * dice
    
def create_criterion(cfg):
    criterion = cfg['name']
    if criterion is None:
        return lambda *x: 0
    elif criterion == 'bce':
        return torch.nn.BCEWithLogitsLoss(
            pos_weight = None if 'pos_weight' not in cfg else torch.tensor(cfg['pos_weight'])[:,None,None,None].to('cuda')
        )
    elif criterion == 'kld':
        class KLDivLossWithLogits(torch.nn.KLDivLoss):
            def __init__(self):
                super().__init__(reduction="batchmean")

            def forward(self, y, t):
                y = F.softmax(y, dim=1)
                loss = super().forward(y, t)

                return loss
        return KLDivLossWithLogits()
    elif criterion == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif criterion == 'dice':
        return DiceLoss()
    elif criterion == 'ce_dice':
        return CE_DiceLoss()
    elif criterion == 'bce_dice':
        return BCE_DiceLoss()
    else:
        raise ValueError(f'Unknown criterion: {criterion}')
    
def list_available_criterions():
    return ['bce', 'kld', 'cross_entropy', 'dice', 'ce_dice', 'bce_dice']
    
def create_metric(cfg):
    metric = cfg['name']
    kwargs = cfg.get('args')
    if kwargs is None:
        kwargs = {}
    if metric == 'kld':
        return torch.nn.KLDivLoss(reduction='batchmean')
    elif metric == 'dice':
        def DiceCoefficient(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-7):
            """
            pred: torch.Tensor with class probabilities. This will be converted to one-hot internally
            target: torch.Tensor with one-hot encoded labels
            """

            # convert to one-hot
            pred_labels = pred.argmax(dim=1)
            pred = torch.nn.functional.one_hot(pred_labels, num_classes=pred.shape[1]).permute((0, 4, 1, 2, 3)).float()
            pred = pred[:, 1:]
            target = target[:, 1:]

            pred = pred.flatten(start_dim=1)
            target = target.flatten(start_dim=1)
            intersection = (pred * target).sum(dim = -1)
            union = pred.sum(dim = -1) + target.sum(dim = -1)
            dice = (2. * intersection + eps) / (union + eps)

            # mean batch
            dice = dice.mean()
            return dice
        return DiceCoefficient
    elif metric == 'dice_binary':
        def Dice(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-7):
            pred = torch.nn.functional.sigmoid(pred)
            pred = pred.flatten(start_dim = 1) > 0.5
            target = target.flatten(start_dim = 1)
            inter = (pred * target).sum(dim=-1)
            union = pred.sum(dim=-1) + target.sum(dim=-1)
            dice = (2. * inter + eps) / (union + eps)

            dice = dice.mean()
            return dice
        return Dice
    else:
        try:
            return create_criterion(cfg)
        except ValueError:
            raise ValueError(f'Unknown metric: {metric}')
