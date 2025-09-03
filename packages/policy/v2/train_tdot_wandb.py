#!/usr/bin/env python3
"""
Training script for Tdot RCFM model with WandB logging
Supports 1500 epochs with batch size 64
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import wandb
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from loaders.tdot_hdf5_dataset import create_dataloader
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM


class WarmupCosineScheduler:
    """Custom scheduler with linear warmup followed by CosineAnnealingWarmRestarts"""
    def __init__(self, optimizer, warmup_epochs, warmup_lr, T_0, T_mult, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_epoch = 0
        
        # Create cosine scheduler for after warmup
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        
    def step(self):
        """Step the scheduler"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing after warmup
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
        
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_epoch(model, dataloader, optimizer, epoch, config, device):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['n_epoch']}")
    
    for i, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        # Create target pose
        target_T = batch['current_T'].clone()
        target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
        
        data = {
            'pc': batch['pc'],
            'current_T': batch['current_T'],
            'target_T': target_T,
            'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
            'T_dot': batch['T_dot']
        }
        
        # Train step
        losses = {}
        loss_dict = model.train_step(data, losses, optimizer)
        
        # Gradient clipping
        if config['training'].get('gradient_clip_norm', 0) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip_norm']
            )
        else:
            grad_norm = 0
        
        epoch_losses.append(loss_dict['loss'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss_dict['loss'],
            'avg_loss': np.mean(epoch_losses[-100:]) if len(epoch_losses) > 0 else 0
        })
        
        # Log to wandb
        if i % config['training']['print_interval'] == 0:
            wandb.log({
                'train/loss': loss_dict['loss'],
                'train/loss_smooth': np.mean(epoch_losses[-100:]) if len(epoch_losses) > 0 else 0,
                'train/gradient_norm': grad_norm,
                'train/step': epoch * len(dataloader) + i,
            })
    
    avg_loss = np.mean(epoch_losses)
    return avg_loss


def validate(model, dataloader, epoch, config, device):
    """Validate the model"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            target_T = batch['current_T'].clone()
            target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
            
            # Get predictions
            v = model.get_latent_vector(batch['pc'].transpose(2, 1))
            T_dot_pred = model.forward(
                batch['current_T'], 
                target_T,
                torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
                v, 
                None
            )
            
            loss = torch.nn.functional.mse_loss(T_dot_pred, batch['T_dot'])
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses)
    
    # Log to wandb
    wandb.log({
        'val/loss': avg_val_loss,
        'epoch': epoch,
    })
    
    return avg_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        default='configs/tdot_rcfm.yml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (overrides config)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    args = parser.parse_args()
    
    print("="*50)
    print("Tdot RCFM Training with WandB")
    print("="*50)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.logdir) / f"tdot_rcfm_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Save config
    with open(log_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize WandB
    if not args.no_wandb:
        wandb_entity = args.wandb_entity or config.get('wandb', {}).get('entity', None)
        wandb.init(
            project=config.get('wandb', {}).get('project_name', 'tdot_rcfm'),
            entity=wandb_entity,
            config=config,
            name=f"tdot_{timestamp}",
            dir=str(log_dir)
        )
        print(f"WandB initialized: {wandb.run.url}")
    else:
        print("WandB disabled")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(config['data']['train'], 'train')
    val_loader = create_dataloader(config['data']['valid'], 'val')
    print(f"Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"Val samples: {len(val_loader.dataset)} ({len(val_loader)} batches)")
    
    # Create model
    print("\nCreating model...")
    latent_feature = DGCNN(config['model']['latent_feature'], output_channels=6)
    velocity_field = vf_FC_vec_motion(
        in_chan=25,
        lat_chan=config['model']['latent_feature']['emb_dims'] * 2,
        out_chan=config['model']['velocity_field']['out_dim'],
        l_hidden=config['model']['velocity_field']['l_hidden'],
        activation=config['model']['velocity_field']['activation'],
        out_activation=config['model']['velocity_field']['out_activation']
    )
    
    model = MotionRCFM(
        velocity_field=velocity_field,
        latent_feature=latent_feature,
        prob_path=config['model']['prob_path'],
        init_dist=config['model']['init_dist'],
        ode_solver=config['model']['ode_solver']
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer'].get('weight_decay', 0.0)
    )
    
    # Create scheduler with warmup
    warmup_epochs = config['training']['scheduler'].get('warmup_epochs', 50)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        warmup_lr=1e-5,  # Start with small LR
        T_0=config['training']['scheduler'].get('T_0', 300),
        T_mult=config['training']['scheduler'].get('T_mult', 2),
        eta_min=config['training']['scheduler'].get('eta_min', 1e-5)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print(f"Epochs: {config['training']['n_epoch']}")
    print(f"Batch size: {config['data']['train']['batch_size']}")
    print(f"Learning rate: {config['training']['optimizer']['lr']}")
    print(f"Warmup epochs: {warmup_epochs}")
    print("="*50 + "\n")
    
    for epoch in range(start_epoch, config['training']['n_epoch']):
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        if not args.no_wandb:
            wandb.log({'train/learning_rate': current_lr, 'epoch': epoch})
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, config, device)
        print(f"\nEpoch {epoch+1}/{config['training']['n_epoch']} - Train Loss: {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % config['training']['val_interval'] == 0:
            val_loss = validate(model, val_loader, epoch, config, device)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = log_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config
                }, save_path)
                print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
                
                if not args.no_wandb:
                    wandb.save(str(save_path))
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_path = log_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
                'config': config
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
        
        # Update learning rate
        scheduler.step()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Logs saved to: {log_dir}")
    if not args.no_wandb:
        wandb.finish()
    print("="*50)


if __name__ == "__main__":
    main()