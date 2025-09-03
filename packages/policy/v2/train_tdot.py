#!/usr/bin/env python3
"""
Training script for Tdot RCFM model
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from loaders.tdot_hdf5_dataset import TdotHDF5Dataset, create_dataloader
from models.motion_rcfm import MotionRCFM


def train_epoch(model, dataloader, optimizer, epoch, config):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    for i, batch in enumerate(dataloader):
        # Move to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        
        # Prepare data for model
        # Create a dummy target pose (current + integrated Tdot)
        target_T = batch['current_T'].clone()
        # Simple forward integration for target
        target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1  # Linear component
        
        data = {
            'pc': batch['pc'],  # [B, N, 3]
            'current_T': batch['current_T'],  # [B, 4, 4]
            'target_T': target_T,  # [B, 4, 4]
            'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,  # Mid-time
            'T_dot': batch['T_dot']  # [B, 6] - this is our target
        }
        
        # Train step
        loss_dict = model.train_step(data, losses, optimizer)
        losses.update(loss_dict['loss'], batch['pc'].size(0))
        
        # Print progress
        if i % config['training']['print_interval'] == 0:
            print(f"Epoch [{epoch}][{i}/{len(dataloader)}] "
                  f"Loss: {losses.avg:.6f}")
    
    return losses.avg


def validate(model, dataloader, epoch, config):
    """Validate the model"""
    model.eval()
    val_losses = AverageMeter()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # Prepare data
            target_T = batch['current_T'].clone()
            target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
            
            data = {
                'pc': batch['pc'],
                'current_T': batch['current_T'],
                'target_T': target_T,
                'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
                'T_dot': batch['T_dot']
            }
            
            # Forward pass through velocity field
            v = model.get_latent_vector(data['pc'].transpose(2, 1))  # Need [B, 3, N]
            T_dot_pred = model.forward(data['current_T'], data['target_T'], 
                                        data['time_t'], v, None)
            loss = torch.nn.functional.mse_loss(T_dot_pred, data['T_dot'])
            val_losses.update(loss.item(), batch['pc'].size(0))
    
    print(f"Validation Epoch [{epoch}] Loss: {val_losses.avg:.6f}")
    return val_losses.avg


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
    args = parser.parse_args()
    
    print("Starting main...")
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("Config loaded")
    
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
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(config['data']['train'], 'train')
    val_loader = create_dataloader(config['data']['valid'], 'val')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    
    # Create latent feature extractor (DGCNN)
    from models.dgcnn import DGCNN
    latent_feature = DGCNN(config['model']['latent_feature'], output_channels=6)
    
    # Create velocity field network
    from models.modules import vf_FC_vec_motion
    velocity_field = vf_FC_vec_motion(
        in_chan=25,  # current(12) + target(12) + time(1) = 25
        lat_chan=config['model']['latent_feature']['emb_dims'] * 2,  # DGCNN features (max+avg pooled)
        out_chan=config['model']['velocity_field']['out_dim'],
        l_hidden=config['model']['velocity_field']['l_hidden'],
        activation=config['model']['velocity_field']['activation'],
        out_activation=config['model']['velocity_field']['out_activation']
    )
    
    # Create RCFM model
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
    
    # Create scheduler if specified
    scheduler = None
    if 'scheduler' in config['training']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max'],
            eta_min=config['training']['scheduler']['eta_min']
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['n_epoch']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['training']['n_epoch']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, config)
        
        # Validate
        if epoch % config['training']['val_interval'] == 0:
            val_loss = validate(model, val_loader, epoch, config)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = log_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            save_path = log_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Logs saved to: {log_dir}")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)


if __name__ == "__main__":
    main()