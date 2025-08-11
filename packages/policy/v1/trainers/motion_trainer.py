import os
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import logging
import torch
from copy import deepcopy
from torch.distributed import destroy_process_group

from losses import get_loss
from utils.metrics import averageMeter
from trainers.schedulers import get_scheduler


class MotionTrainer:
    """Trainer for motion planning model training"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.best_model_criteria = self.cfg.get('best_model_criteria', None)
        self.ddp = self.cfg.get('ddp', False)
        self.global_rank = int(os.environ['RANK']) if self.ddp else 0
        self.losses = [get_loss(cfg_loss) for cfg_loss in training_cfg['losses'].values()] if 'losses' in training_cfg else []
        
    def train(self, model, optimizer, dataloaders, logger=None, logdir=None):
        """Main training loop for motion planning model"""
        
        # Set up scheduler  
        scheduler = None  # 일단 스케줄러 비활성화
        self.log_dir = logdir
        
        # Extract dataloaders
        train_loader = dataloaders['train']
        val_loader = dataloaders.get('valid', None)
        
        n_epochs = self.cfg.get('n_epoch', 1000)
        print_interval = self.cfg.get('print_interval', 50)
        val_interval = self.cfg.get('val_interval', 200)
        save_interval = self.cfg.get('save_interval', 1000)
        
        best_loss = float('inf')
        loss_meter = averageMeter()
        
        model.train()
        
        for epoch in range(n_epochs):
            loss_meter.reset()
            
            for batch_idx, data in enumerate(train_loader):
                # Move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.device)
                
                # Training step
                loss_dict = model.train_step(data, self.losses, optimizer)
                
                # Update metrics
                train_loss = loss_dict.get('loss/train_loss_', 0.0)
                loss_meter.update(train_loss)
                
                # Logging
                if (batch_idx + 1) % print_interval == 0:
                    print(f'Epoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                          f'Loss: {train_loss:.6f}, Avg Loss: {loss_meter.avg:.6f}')
                    
                    # Log to wandb/tensorboard  
                    logger.process_iter_train({'loss': train_loss})
                    for key, value in loss_dict.items():
                        logger.d_train[key] = value
                    logger.summary_train(epoch * len(train_loader) + batch_idx)
            
            # Validation
            if (epoch + 1) % val_interval == 0:
                val_loss = self.validate(model, val_loader, epoch, logger)
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(model, optimizer, epoch, 'best_normalized', logdir)
                    print(f'New best normalized model saved at epoch {epoch+1} with loss: {val_loss:.6f}')
            
            # Regular model saving
            if (epoch + 1) % save_interval == 0:
                self.save_model(model, optimizer, epoch, f'epoch_{epoch+1}_normalized', logdir)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
        
        print(f'Training completed! Best validation loss: {best_loss:.6f}')
        
        return model, optimizer
        
    def validate(self, model, val_loader, epoch, logger):
        """Validation loop"""
        model.eval()
        val_loss_meter = averageMeter()
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                # Move data to device
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(self.device)
                
                # Forward pass only (no optimizer step)
                current_T = data['current_T']
                target_T = data['target_T']
                time_t = data['time_t']
                pointcloud = data['pointcloud']
                T_dot_target = data['T_dot']
                
                # Encode pointcloud
                if pointcloud.dim() == 3 and pointcloud.shape[1] < pointcloud.shape[2]:
                    v = model.latent_feature(pointcloud)
                else:
                    pointcloud_dgcnn = pointcloud.transpose(1, 2)
                    v = model.latent_feature(pointcloud_dgcnn)
                
                # Forward pass
                T_dot_pred = model.forward(current_T, target_T, time_t, v)
                
                # Compute validation loss
                val_loss = 0
                if self.losses:
                    for loss_fn in self.losses:
                        loss_val = loss_fn(T_dot_pred, T_dot_target)
                        val_loss += loss_val.item()
                else:
                    # Fallback to MSE loss if no losses defined
                    import torch.nn as nn
                    mse_loss = nn.MSELoss()
                    val_loss = mse_loss(T_dot_pred, T_dot_target).item()
                
                val_loss_meter.update(val_loss)
        
        model.train()
        
        # Log validation results
        d_val_results = {'validation/loss_': val_loss_meter.avg}
        logger.logging(epoch, d_val_results)
        print(f'Validation Loss: {val_loss_meter.avg:.6f}')
        
        return val_loss_meter.avg
    
    def save_model(self, model, optimizer, epoch, name, log_dir):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
        }
        
        save_path = os.path.join(log_dir, f'model_{name}.pth')
        torch.save(checkpoint, save_path)
        print(f'Model saved: {save_path}')
    
    def load_model(self, model, optimizer, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch = checkpoint.get('epoch', 0)
        print(f'Model loaded from {checkpoint_path}, epoch: {epoch}')
        return epoch

