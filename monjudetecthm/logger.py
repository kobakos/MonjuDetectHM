import os
import shutil
import datetime
from pathlib import Path

import yaml
import torch
import numpy as np
from timm.utils.model_ema import ModelEmaV2

from .utils import sigmoid

class EmaCalculator:
    def __init__(self, alpha=0.99, direct_update_steps=100):
        self.alpha = alpha
        self.ema = None
        self.direct_update_steps = direct_update_steps

        self.history = []
        self.require_adjust = (direct_update_steps > 0)

    def update(self, value):
        if self.require_adjust:
            self.history.append(value)
            if len(self.history) >= self.direct_update_steps:
                self.require_adjust = False
            self.ema = np.average(self.history, weights = self.alpha ** np.arange(len(self.history)-1, -1, -1))
            return self.ema
        
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * self.ema + (1 - self.alpha) * value
        return self.ema
    
    def __str__(self):
        return f'{self.ema:.5f}'
    
    def __float__(self):
        return self.ema

class weights_logger():
    def __init__(self, path = Path('weights')):
        self.path = Path(path)
        os.makedirs(path, exist_ok=True)

    def save(self, model, f):
        if isinstance(model, ModelEmaV2):
            torch.save(model.module.state_dict(), self.path/f)
        else:
            torch.save(model.state_dict(), self.path/f)

    def load(self, model, f):
        model.load_state_dict(torch.load(self.path/f))
        return model
    
class Logger():
    # creates a new folder for this iteration and initializes the wandb run
    def __init__(self, cfg, experiment_path, disable_wandb=False, config_path = Path('./'), metric_init = 0):
        self.experiment_path = Path(experiment_path)
        #wandb.init(project=cfg['project_name'], entity='kobakos')
        self.iteration_name = self._name_iteration(cfg['model']['name'])

        os.makedirs(self.experiment_path/self.iteration_name, exist_ok=True)
        #shutil.copy(config_path/'config.yml', self.experiment_path/self.iteration_name/'config.yml')
        with open(self.experiment_path/self.iteration_name/'config.yml', 'w') as f:
            yaml.dump(cfg, f)
        
        self.disable_wandb = disable_wandb
        if not disable_wandb:
            wandb.init(project=cfg['project_name'], entity='kobakos', name=self.iteration_name, config=cfg)
            wandb.run.log_code(".")

        self.best_metric = metric_init
        self.metric_init = metric_init
        self.time_since_best = 0
        self.experiment_path = experiment_path
        
        self.wl = weights_logger(self.experiment_path/self.iteration_name/'weights')

    # takes the experiment path and creates a new directory for the new experiment by incrementing the experiment number
    def _name_iteration(self, path_prefix = ''):
        if len(os.listdir(self.experiment_path)) == 0:
            exp_iteration = 0
        else:
            exp_iteration = max(map(lambda p: int(p.split('_')[0]), os.listdir(self.experiment_path)))
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        return f"{exp_iteration + 1}_{path_prefix}_{now:%Y%m%d}"
    
    def log(self, values_to_log: dict):
        if not self.disable_wandb:
            wandb.log(values_to_log)
    def log_heatmap_preds(self, image, target, output, fold, epoch, name='val_pred'):
        for c in range(target.shape[1]):
            idx = np.unravel_index(np.argmax(target[:, c], axis=None), target[:, c].shape)
            image_to_log = np.concatenate([
                image[idx[0], 0, idx[1]] / 6 + 0.5,
                target[idx[0], c, idx[1]],
                sigmoid(output[idx[0], c, idx[1]]),
            ], axis=1).clip(0, 1)[:,:,None]
            self.log({
                name: wandb.Image(image_to_log, caption=f'fold_{fold}_epoch_{epoch}_class_{c}')
            })

    def update_best_metric(self, metric, mode='max'):
        if mode == 'max':
            if metric > self.best_metric:
                self.best_metric = metric
                self.time_since_best = 0
            else:
                self.time_since_best += 1
        elif mode == 'min':
            if metric < self.best_metric:
                self.best_metric = metric
                self.time_since_best = 0
            else:
                self.time_since_best += 1
        else:
            raise ValueError(f'Unknown mode: {mode}')
        
    def reset_best_metric(self):
        self.best_metric = self.metric_init
        self.time_since_best = 0
        
    def early_stop(self, patience=5):
        """
        this has to be called after update_best_metric
        """
        if self.time_since_best > patience:
            return True
        return False

    def save_model(self, model, fold=None, epoch=None, best_only=False): 
        """
        this has to be called after update_best_metric
        """
        os.makedirs(self.experiment_path/self.iteration_name/'weights'/f'fold_{fold}', exist_ok=True)
        if best_only:
            if self.time_since_best == 0:
                print('saving best model...')
                self.wl.save(model, f"fold_{fold}/best.pth")
        else:
            self.wl.save(model, f'fold_{fold}/epoch_{epoch}.pth')
