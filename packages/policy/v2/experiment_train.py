#!/usr/bin/env python3
"""
Training experiment with small dataset
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import create_dataloader
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

class TrainingExperiment:
    def __init__(self, config_path='configs/tdot_rcfm.yml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override for small experiment
        self.config['data']['train']['max_trajectories'] = 10
        self.config['data']['train']['batch_size'] = 4
        self.config['data']['train']['num_points'] = 512
        self.config['data']['valid']['max_trajectories'] = 5
        self.config['data']['valid']['batch_size'] = 4
        self.config['data']['valid']['num_points'] = 512
        self.config['training']['n_epoch'] = 50
        self.config['training']['print_interval'] = 5
        self.config['training']['val_interval'] = 10
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create log directory
        self.log_dir = Path('experiments') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.log_dir / 'config.yml', 'w') as f:
            yaml.dump(self.config, f)
        
        self.train_losses = []
        self.val_losses = []
        
    def create_model(self):
        """Create model from config"""
        print("Creating model...")
        
        latent_feature = DGCNN(self.config['model']['latent_feature'], output_channels=6)
        
        velocity_field = vf_FC_vec_motion(
            in_chan=25,
            lat_chan=self.config['model']['latent_feature']['emb_dims'] * 2,
            out_chan=self.config['model']['velocity_field']['out_dim'],
            l_hidden=self.config['model']['velocity_field']['l_hidden'],
            activation=self.config['model']['velocity_field']['activation'],
            out_activation=self.config['model']['velocity_field']['out_activation']
        )
        
        model = MotionRCFM(
            velocity_field=velocity_field,
            latent_feature=latent_feature,
            prob_path=self.config['model']['prob_path'],
            init_dist=self.config['model']['init_dist'],
            ode_solver=self.config['model']['ode_solver']
        )
        
        model = model.to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        
        return model
    
    def create_dataloaders(self):
        """Create train and validation dataloaders"""
        print("Creating dataloaders...")
        
        train_loader = create_dataloader(self.config['data']['train'], 'train')
        val_loader = create_dataloader(self.config['data']['valid'], 'val')
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model, dataloader, optimizer, epoch):
        """Train for one epoch"""
        model.train()
        epoch_losses = []
        
        for i, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
            
            # Create target pose
            target_T = batch['current_T'].clone()
            target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
            
            data = {
                'pc': batch['pc'],
                'current_T': batch['current_T'],
                'target_T': target_T,
                'time_t': torch.ones(batch['pc'].size(0), 1).to(self.device) * 0.5,
                'T_dot': batch['T_dot']
            }
            
            losses = {}
            loss_dict = model.train_step(data, losses, optimizer)
            epoch_losses.append(loss_dict['loss'])
            
            if i % self.config['training']['print_interval'] == 0:
                print(f"  Batch [{i}/{len(dataloader)}] Loss: {loss_dict['loss']:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        return avg_loss
    
    def validate(self, model, dataloader):
        """Validate model"""
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in batch.items()}
                
                target_T = batch['current_T'].clone()
                target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
                
                # Get predictions
                v = model.get_latent_vector(batch['pc'].transpose(2, 1))
                T_dot_pred = model.forward(
                    batch['current_T'], 
                    target_T,
                    torch.ones(batch['pc'].size(0), 1).to(self.device) * 0.5,
                    v, 
                    None
                )
                
                loss = torch.nn.functional.mse_loss(T_dot_pred, batch['T_dot'])
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def run_training(self):
        """Run the full training experiment"""
        print("\n" + "="*50)
        print("Starting Training Experiment")
        print("="*50)
        
        # Create model and dataloaders
        model = self.create_model()
        train_loader, val_loader = self.create_dataloaders()
        
        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['optimizer']['lr'],
            weight_decay=self.config['training']['optimizer'].get('weight_decay', 0.0)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['n_epoch'],
            eta_min=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['n_epoch']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['n_epoch']}")
            print("-" * 30)
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.6f}")
            
            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss = self.validate(model, val_loader)
                self.val_losses.append(val_loss)
                print(f"Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, self.log_dir / 'best_model.pth')
                    print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
            
            # Update learning rate
            scheduler.step()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.save_metrics()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"Results saved to: {self.log_dir}")
        print("="*50)
        
        return model, best_val_loss
    
    def save_metrics(self):
        """Save training metrics"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'n_epochs': self.config['training']['n_epoch'],
                'batch_size': self.config['data']['train']['batch_size'],
                'learning_rate': self.config['training']['optimizer']['lr'],
                'train_trajectories': self.config['data']['train']['max_trajectories'],
            }
        }
        
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def analyze_results(self):
        """Analyze and print training results"""
        print("\n📊 Training Analysis:")
        print("-" * 30)
        
        if len(self.train_losses) > 0:
            print(f"Initial Train Loss: {self.train_losses[0]:.6f}")
            print(f"Final Train Loss: {self.train_losses[-1]:.6f}")
            print(f"Training Improvement: {(1 - self.train_losses[-1]/self.train_losses[0])*100:.1f}%")
        
        if len(self.val_losses) > 0:
            print(f"Best Val Loss: {min(self.val_losses):.6f}")
            print(f"Final Val Loss: {self.val_losses[-1]:.6f}")
        
        # Check for overfitting
        if len(self.val_losses) > 2:
            recent_val = self.val_losses[-3:]
            if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
                print("⚠️  Warning: Validation loss is increasing (possible overfitting)")
            else:
                print("✅ Model is still improving")


if __name__ == "__main__":
    print("🚀 Starting Tdot Training Experiment")
    
    experiment = TrainingExperiment()
    model, best_loss = experiment.run_training()
    experiment.analyze_results()
    
    print("\n✨ Experiment complete!")