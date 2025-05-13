import numpy as np

import torch
import torch.nn.functional as F



from src.component_factory import create_criterion
#from models.detr.prepare_detr import prepare_czii_setcriterion
def from_cfg(cfg):
    if cfg["name"] == "FocalTverskyLoss":
        return FocalTverskyLoss()
    elif cfg["name"] == "WeightedFocalLoss":
        return WeightedFocalLoss()
    elif cfg["name"] == "WingLoss":
        return WingLoss()
    #elif cfg['name'] == 'detr':
    #    return prepare_czii_setcriterion(**cfg['args'])
    elif cfg['name'] == 'normalized_bce':
        return NormalizedBCE(
            eps = cfg.get('eps', 0),
            pos_weight = None if 'pos_weight' not in cfg else torch.tensor(cfg['pos_weight'])[None, :,None,None,None].to('cuda')
        )
    elif cfg['name'] == 'bce_fbeta':
        return BlendedLoss(
            SoftFbetaLoss(beta=cfg.get('beta', 4)),
            torch.nn.BCEWithLogitsLoss(pos_weight = None if 'pos_weight' not in cfg else torch.tensor(cfg['pos_weight']).to('cuda'))
        )
    elif cfg['name'] == 'bce':
        return CZIIBceLoss(
            pos_weight = None if 'pos_weight' not in cfg else torch.tensor(cfg['pos_weight'])[:,None,None,None].to('cuda'),
            weight=None if 'class_weight' not in cfg else cfg['class_weight']
        )
    elif cfg['name'] == 'dann_bce':
        return DANNLoss(from_cfg(cfg['classifier']), cfg['dann_weight'])
    
    elif cfg['name'] == 'bce_w_offset':
        return BceWithOffset(
            pos_weight = None if 'pos_weight' not in cfg else torch.tensor(cfg['pos_weight'])[:,None,None,None].to('cuda'),
            weight=None if 'class_weight' not in cfg else cfg['class_weight'],
            bce_weight=cfg.get('bce_weight', 1.0),
            offset_weight=cfg.get('offset_weight', 1.0),
            threshold=cfg.get('threshold', 0.3),
            class_weight = cfg.get('class_weight', None),
            peak_detection_kernel_size = cfg.get('peak_detection_kernel_size', 3),
            min_dist_multiplier = cfg.get('min_dist_multiplier', 1.0),
            classes = cfg.get('classes', ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']),
            warmup_method = cfg.get('warmup_method', None),
            warmup_params = cfg.get('warmup_params', None),
        )


    else:
        return create_criterion(cfg)

class DANNLoss(torch.nn.Module):
    def __init__(self, main_criterion, dann_weight=1):
        super().__init__()
        self.classifier_criterion = main_criterion
        self.domain_criterion = torch.nn.BCEWithLogitsLoss()
        self.dann_weight = dann_weight
    def forward(self, model_output, target):
        classifier_output, domain_output = model_output
        target["domain"] = target["domain"].squeeze()
        if (target['domain'] == 0).all():
            return self.classifier_criterion(classifier_output, target['heatmap'])
        classifier_loss = self.classifier_criterion(
            classifier_output[target["domain"]==1],
            {'heatmap': target['heatmap'][target["domain"]==1]}
        )
        domain_loss = self.domain_criterion(domain_output, target['domain'][:, None])
        return classifier_loss + self.dann_weight * domain_loss

class BlendedLoss(torch.nn.Module):
    def __init__(self, criterion1, criterion2, alpha=0.5):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.alpha = alpha
    def forward(self, y, t):
        return self.alpha * self.criterion1(y, t) + (1 - self.alpha) * self.criterion2(y, t)
    
class FocalLossPosWeight(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super(FocalLossPosWeight, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs_clamped = torch.clamp(probs, min=1e-6, max=1-1e-6)  # Avoid log(0)
        
        # Compute BCE components
        bce_loss = targets * torch.log(probs_clamped) + (1 - targets) * torch.log(1 - probs_clamped)
        
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            pos_weighted_bce = self.pos_weight * targets * torch.log(probs_clamped) + (1 - targets) * torch.log(1 - probs_clamped)
        else:
            pos_weighted_bce = bce_loss
        
        # Compute focal loss modulation
        focal_modulation = (1 - probs_clamped).pow(self.gamma)
        
        # Combine terms
        loss = -self.alpha * focal_modulation * pos_weighted_bce
        return loss.mean()
    
class SoftFbetaLoss(torch.nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta
    def forward(self, y, t):
        y = torch.sigmoid(y)
        tp = (y * t).sum()
        fp = ((1 - t) * y).sum()
        fn = (t * (1 - y)).sum()
        return 1 - (1 + self.beta**2) * tp / ((1 + self.beta**2) * tp + self.beta**2 * fn + fp)
    
class CZIIBceLoss(torch.nn.Module):
    def __init__(self, pos_weight = None, weight = None):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction = 'none'
        )
        if weight is not None:
            self.weight = torch.Tensor(weight).to('cuda')
        else:
            self.weight = None
    def forward(self, y, t):
        loss = self.bce(y, t['heatmap'])
        loss = loss.mean(dim=(0, 2, 3, 4))
        if self.weight is not None:
            loss = loss * self.weight
        return loss.mean()

class BceWithOffset(torch.nn.Module):
    def __init__(self, pos_weight = None, weight = None, bce_weight=1.0, offset_weight=1.0, **kwargs):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction = 'none'
        )
        if weight is not None:
            self.weight = torch.Tensor(weight).to('cuda')
        else:
            self.weight = None
        self.offset_loss = HmapRegHeadLoss(**kwargs)
        self.bce_weight = bce_weight
        self.offset_weight = offset_weight
    def forward(self, y, t):
        loss = self.bce(y[0], t['heatmap'])
        loss = loss.mean(dim=(0, 2, 3, 4))
        if self.weight is not None:
            loss = loss * self.weight
        loss = loss.mean()
        loss *= self.bce_weight
        loss += self.offset_weight * self.offset_loss(y, t)
        return loss

import src.constants
class HmapRegHeadLoss(torch.nn.Module):
    def __init__(self,
            threshold=0.3,
            class_weight = None,
            peak_detection_kernel_size = 3,
            min_dist_multiplier = 1.0,
            pixel_spacing = 10.012444196428572,
            classes = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle'],
            warmup_method = None,
            warmup_params = None
        ):
        super().__init__()
        self.th = threshold

        self.classes = classes
        if isinstance(min_dist_multiplier, (int, float)):
            self.min_dist_multiplier = [min_dist_multiplier]*len(classes)
        else:
            assert len(min_dist_multiplier) == len(classes), "min_dist_multiplier should be a scalar or a list of length equal to the number of classes"
            self.min_dist_multiplier = min_dist_multiplier

        self.pixel_spacing = pixel_spacing

        self.reg_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.weight = class_weight
        self.particle_radius = constants.particle_radius
        self.particle_radius = {k: v / self.pixel_spacing for k, v in self.particle_radius.items()}# convert to pixel

        assert peak_detection_kernel_size % 2 == 1, "peak_detection_kernel_size should be odd"
        self.peak_detection_kernel_size = peak_detection_kernel_size

        
        print(warmup_method)
        if warmup_method == 'linear':
            def warmup_linear(epoch):
                if epoch < warmup_params['n_epochs']:
                    return epoch / warmup_params['n_epochs']
                else:
                    return 1
            self.warmup_func = warmup_linear
        elif warmup_method == 'sigmoid':
            def warmup_sigmoid(epoch):
                sigm = 1 / (1 + np.exp(- warmup_params['k'] * epoch))
                return 2 * sigm - 1
            self.warmup_func = warmup_sigmoid
        else:
            raise ValueError("warmup_method should be either 'linear' or 'sigmoid'")

    def forward(self, model_out, target):
        """
        model_out: tuple of torch.Tensor. The output of the model. 
            The first element is the heatmap and the second element is the tuple of regression maps.
            (heatmap, (reg_c1, reg_c2, ...))
            heatmap: torch.Tensor with shape (n, c, d, h, w)
            reg_maps: tuple of torch.Tensor with shape (n, 3, d, h, w)
        target: dictionary. For this loss, it should contain:
            'heatmap': torch.Tensor. The target heatmap
            'points': dict. should have class_str as key and the point(torch.Tensor with shape (n, 4)) as values. should be pixel coordinates.
        """
        hmap, reg_maps = model_out
        with torch.no_grad():
            pooled = F.max_pool3d(hmap, kernel_size=self.peak_detection_kernel_size, stride=1, padding=self.peak_detection_kernel_size//2)
            pred_peaks = (pooled==hmap) & (hmap > self.th)
            objects_to_train, target_offsets = self.filter_positive_dist(pred_peaks, target['points'])
        total_loss = 0
        for ci, class_str in enumerate(self.classes):
            if len(objects_to_train[class_str]) == 0:
                continue
            #print(target_offsets[class_str])
            reg_loss = self.reg_loss(
                #reg_maps[ci][
                reg_maps[
                    objects_to_train[class_str][:, 0],# batch
                    :,  # zyx
                    objects_to_train[class_str][:, 1],
                    objects_to_train[class_str][:, 2],
                    objects_to_train[class_str][:, 3]
                ],
                target_offsets[class_str]
            )
            if self.weight is not None:
                reg_loss = reg_loss * self.weight[ci]
            total_loss += reg_loss

        if self.warmup_func is not None:
            total_loss *= self.warmup_func(target['epoch'])

        return total_loss
            
    def to_pixel(self, points):
        return (points / self.pixel_spacing)[:, [2, 1, 0]]# convert to zyx pixel coordinates

    def filter_positive_dist(self, pred_peaks, target_points):
        objects_to_train = {}
        target_offsets = {}
        for ci, class_str in enumerate(self.classes):
            if class_str not in target_points or len(target_points[class_str]) == 0:
                objects_to_train[class_str] = torch.zeros(0, 4, dtype=torch.long)
                continue
            pred_coords = torch.stack(torch.where(pred_peaks[:, ci]), dim=1)# 
            diff = pred_coords[:, None, 1:] - target_points[class_str][None, :, 1:] # (num_detected_objects, num_target_points, 3)
            distances = torch.norm(diff, dim=2)
            min_dist_idx = torch.argmin(distances, dim=1)# (num_detected_objects,)
            dist_thresh = self.particle_radius[class_str] * self.min_dist_multiplier[ci]
            min_dist_idx[distances[torch.arange(len(pred_coords)), min_dist_idx] > dist_thresh] = -1
            # only keep the detected objects that are close to the target points
            objects_to_train[class_str] = pred_coords[min_dist_idx != -1]
            targeted_points = target_points[class_str][min_dist_idx[min_dist_idx != -1]]
            #print(targeted_points[:, 1:], pred_coords[min_dist_idx != -1][:, 1:])
            # discard the batch coordinates
            target_offsets[class_str] = targeted_points[:, 1:] - pred_coords[min_dist_idx != -1][:, 1:]

            target_offsets[class_str] = target_offsets[class_str] / self.particle_radius[class_str]
        return objects_to_train, target_offsets


    def filter_positive_hmap(self, detected_objects, target_hmap):
        return detected_objects & (target_hmap > self.th)

if __name__ == '__main__':
    classes = ['c1', 'c2']
    reg_maps = [torch.rand(2, 3, 10, 10, 10) for _ in classes]
    target = {
        'heatmap': torch.rand(2, 2, 10, 10, 10),
        'points': {
            'c1': torch.Tensor([
                [0, 0, 0, 0],
                [0, 0, 5, 5]
            ]),
            'c2': torch.Tensor([
                [1, 0, 0, 0],
                [1, 0, 5, 5]
            ]),
        }
    }
    hmap = torch.rand(2, 2, 10, 10, 10)
    hmap[0, 0, 0, 0, 0] = 100
    hmap[1, 1, 0, 5, 5] = 100
    loss = HmapRegHeadLoss(classes=classes)
    print(loss((hmap, reg_maps), target))
    
class NormalizedBCE(torch.nn.Module):
    def __init__(self, eps=0, pos_weight = None):
        super().__init__()
        
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight
        )
        self.eps = eps
    def forward(self, y, t):
        t, num_points = t
        bce = self.bce(y, t)
        if num_points == 0 and self.eps==0:
            return bce
        return bce / num_points

class HeatmapWeightedBCE(torch.nn.Module):
    def __init__(self, range = (0.05, 1)):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(
            reduction='none'
        )
        self.range = range
    def forward(self, y, t):
        weight = t * (self.range[1] - self.range[0]) + self.range[0]
        return (self.bce(y, t) * weight).mean()

class WeightedFocalLoss(torch.nn.Module):
    def __init__(self, weights = [1], alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = torch.tensor(weights).float().cuda()
    def forward(self, pred, target):
        """
        pred: (N, C, D, H, W)
        target: (N, C, D, H, W)
        """
        pred_sig = torch.sigmoid(pred)
        pred_logsig = torch.nn.functional.logsigmoid(pred)# use this for numerical stability
        pred_log1nsig = -torch.nn.functional.softplus(pred)
        #breakpoint()
        loss = -torch.sum(torch.where(
            target == 1,
            (1 - pred_sig)**self.alpha * pred_logsig,
            (1 - target)**self.beta * pred_sig**self.alpha * pred_log1nsig,
        ), dim=(0, 2, 3, 4))# dimensions excluding channel (class) dimension
        loss = loss * self.class_weights
        loss = loss.mean()
        return loss
    
class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, epsilon=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        ones = torch.ones_like(y_true)
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true

        num = torch.sum(p0 * g0, dim=(0, 2, 3, 4))
        den = num + self.alpha * torch.sum(p0 * g1, dim=(0, 2, 3, 4)) + self.beta * torch.sum(p1 * g0, dim=(0, 2, 3, 4)) + self.epsilon

        T = torch.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = y_true.shape[1]
        return Ncl - T

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, epsilon=1e-3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        # Clip predictions to prevent log(0) errors
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        
        # Calculate true positives, false negatives, and false positives
        true_pos = torch.sum(y_true * y_pred, dim=(1, 2, 3, 4))
        false_neg = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3, 4))
        false_pos = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3, 4))
        
        # Calculate the Tversky index
        tversky_index = (true_pos + self.epsilon) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.epsilon)
        
        # Apply the focal term
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)
        
        # Return the mean loss over the batch
        return focal_tversky_loss.mean()
from math import log
class WingLoss(torch.nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
    
    def forward(self, preds, targets):
        x = preds - targets
        C = self.omega * (1 - log(1 + self.omega / self.epsilon))
        absolute_x = torch.abs(x)
        losses = torch.where(
            absolute_x < self.omega,
            self.omega * torch.log(1 + absolute_x / self.epsilon),
            absolute_x - C
        )
        return losses.mean()
