import numpy as np
from utils.metrics import averageMeter
import copy, os

import wandb

class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer, endwith=[], ddp=False):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.endwith = endwith
        self.train_loss_meter = averageMeter()
        self.val_loss_meter = averageMeter()
        self.d_train = {}
        self.d_val = {}
        self.ddp = ddp
        if self.ddp:
            self.global_rank = int(os.environ["RANK"])
        self.wandb_upload = (
            (~self.ddp) or (
                (self.ddp) and (self.global_rank == 0)
                )
        ) and (wandb.run is not None)
        
    def process_iter_train(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i):
        self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                if self.wandb_upload:
                    wandb.log({key: val}, step=i)
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)
                    if self.wandb_upload:
                        images = wandb.Image(val)
                        wandb.log({key: images}, step=i)
            if key.endswith('#') and ('#' in self.endwith):
                if val is not None:
                    self.writer.add_figure(key, val, i)
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
            if key.endswith('$') and ('$' in self.endwith): # only wandb
                if val is not None:
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
        result = self.d_train
        self.d_train = {}
        return result

    def process_iter_val(self, d_result):
        self.val_loss_meter.update(d_result['loss'])
        self.d_val = d_result

    def summary_val(self, i):
        self.d_val['loss/val_loss_'] = self.val_loss_meter.avg
        l_print_str = [f'Iter [{i:d}]']
        for key, val in self.d_val.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                l_print_str.append(f'{key}: {val:.4f}')
                if self.wandb_upload:
                    wandb.log({key: val}, step=i)
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)
                    if self.wandb_upload:
                        images = wandb.Image(val)
                        wandb.log({key: images}, step=i)
            if key.endswith('#') and ('#' in self.endwith):
                if val is not None:
                    self.writer.add_figure(key, val, i)
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
            if key.endswith('$') and ('$' in self.endwith): # only wandb
                if val is not None:
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
            if key.endswith('%') and ('%' in self.endwith):
                self.writer.add_mesh(key, vertices=val[0], faces=val[1], colors=val[2], global_step=i)
        print_str = ' '.join(l_print_str)

        result = self.d_val
        result['print_str'] = print_str
        self.d_val = {}
        return result
    
    def logging(self, i, d_results):
        for key, val in d_results.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                if self.wandb_upload:
                    wandb.log({key: val}, step=i)
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)
                    if self.wandb_upload:
                        images = wandb.Image(val)
                        wandb.log({key: images}, step=i)
            if key.endswith('#') and ('#' in self.endwith):
                if val is not None:
                    self.writer.add_figure(key, val, i)
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
            if key.endswith('$') and ('$' in self.endwith): # only wandb
                if val is not None:
                    if self.wandb_upload:
                        wandb.log({key: val}, step=i)
            if key.endswith('%') and ('%' in self.endwith):
                self.writer.add_mesh(key, vertices=val[0], faces=val[1], colors=val[2], global_step=i)
                    