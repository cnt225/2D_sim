"""
SE(3) Riemannian Flow Matching Training Script

Complete training script for SE(3) RFM models with:
- Multi-GPU distributed training support
- Wandb integration for experiment tracking
- Comprehensive evaluation and visualization
- Checkpoint management and resuming
- Adaptive loss weighting
"""

import argparse
import os
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import wandb

# Local imports
from models.se3_rfm import SE3RFM
from loaders import get_dataloader
from losses import MultiTaskLoss, compute_velocity_metrics
from trainers.optimizers import get_optimizer
from trainers.schedulers import get_scheduler
from utils.se3_utils import SE3Utils
from evaluation.evaluator import SE3RFMEvaluator


class SE3RFMTrainer:
    """
    Complete trainer for SE(3) Riemannian Flow Matching models
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging and wandb
        self.setup_logging()
        self.setup_wandb()
        
        # Setup model and training components
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.loss_history = {}
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Evaluation
        self.evaluator = SE3RFMEvaluator(self.model, self.device)
        
        self.log(f"Initialized SE3RFM Trainer on {self.device}")
    
    def setup_directories(self):
        """Setup training directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get('experiment_name', f'se3_rfm_{timestamp}')
        
        self.exp_dir = Path(self.config.get('exp_dir', './experiments')) / exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.vis_dir = self.exp_dir / 'visualizations'
        
        # Create directories (only on rank 0)
        if self.rank == 0:
            for directory in [self.exp_dir, self.checkpoint_dir, self.log_dir, self.vis_dir]:
                directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        if self.rank == 0:
            logging.basicConfig(
                filename=self.log_dir / 'training.log',
                format='%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y/%m/%d %I:%M:%S %p',
                level=logging.INFO
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.rank == 0 and self.config.get('use_wandb', True):
            wandb_config = self.config.get('wandb', {})
            
            wandb.init(
                project=wandb_config.get('project', 'se3-rfm'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', self.exp_dir.name),
                config=self.config,
                dir=str(self.exp_dir),
                tags=wandb_config.get('tags', ['se3', 'rfm', 'robotics'])
            )
            
            # Watch model (will be set up after model creation)
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def setup_model(self):
        """Setup SE3RFM model"""
        model_config = self.config['model']
        
        # Calculate input dimension for velocity field
        pc_output_dim = model_config['point_cloud_encoder']['output_dim']
        geom_output_dim = model_config['geometry_encoder']['output_dim']
        se3_output_dim = model_config['se3_encoder']['output_dim']
        
        velocity_field_input_dim = (
            se3_output_dim * 2 +  # current + target pose features
            pc_output_dim +       # point cloud features
            geom_output_dim +     # geometry features
            1                     # time
        )
        
        model_config['velocity_field_config']['input_dim'] = velocity_field_input_dim
        
        # Create model
        self.model = SE3RFM(
            point_cloud_config=model_config['point_cloud_encoder'],
            geometry_config=model_config['geometry_encoder'],

            velocity_field_config=model_config['velocity_field_config'],
            prob_path=model_config.get('prob_path', 'OT'),
            init_dist=model_config.get('init_dist', {'type': 'uniform'}),
            ode_solver=model_config.get('ode_solver', {'type': 'rk4', 'n_steps': 20})
        ).to(self.device)
        
        # Setup distributed training
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Watch model with wandb
        if self.use_wandb:
            wandb.watch(self.model, log='all', log_freq=1000)
        
        self.log(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self):
        """Setup data loaders"""
        data_config = self.config['data']
        
        # Training data loader
        self.train_loader = get_dataloader({
            "data_root": data_config['train']['data_root'],
            "batch_size": data_config['train']['batch_size'],
            "shuffle": True,
            "num_workers": data_config['train'].get('num_workers', 4),
            "max_trajectories": data_config['train'].get('max_trajectories', None),
            "augment_data": data_config['train'].get('augment_data', True),
            "use_bsplined": data_config['train'].get('use_bsplined', True),
            "dataset": "se3_trajectory"
        })
        
        # Validation data loader
        self.val_loader = get_dataloader({
            "data_root": data_config['val']['data_root'],
            "batch_size": data_config['val']['batch_size'],
            "shuffle": False,
            "num_workers": data_config['val'].get('num_workers', 2),
            "max_trajectories": data_config['val'].get('max_trajectories', None),
            "augment_data": False,  # No augmentation for validation
            "use_bsplined": data_config['val'].get('use_bsplined', True),
            "dataset": "se3_trajectory"
        })
        
        self.log(f"Data setup complete. Train: {len(self.train_loader)} batches, Val: {len(self.val_loader)} batches")
    
    def setup_training(self):
        """Setup training components"""
        training_config = self.config['training']
        
        # Loss function
        loss_config = training_config.get('loss', {})
        self.criterion = MultiTaskLoss(**loss_config).to(self.device)
        
        # Optimizer
        optimizer_config = training_config['optimizer']
        self.optimizer = get_optimizer(
            self.model.parameters(),
            **optimizer_config
        )
        
        # Learning rate scheduler
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config:
            self.scheduler = get_scheduler(self.optimizer, **scheduler_config)
        else:
            self.scheduler = None
        
        # Training parameters
        self.num_epochs = training_config.get('num_epochs', 1000)
        self.log_interval = training_config.get('log_interval', 50)
        self.val_interval = training_config.get('val_interval', 500)
        self.save_interval = training_config.get('save_interval', 1000)
        self.vis_interval = training_config.get('vis_interval', 2000)
        
        self.log("Training setup complete")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_losses = {}
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss_dict, metrics_dict = self.train_step(batch)
            
            # Accumulate losses and metrics
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
            
            for key, value in metrics_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Logging
            if batch_idx % self.log_interval == 0:
                self.log_training_step(batch_idx, loss_dict, metrics_dict)
            
            # Validation
            if self.global_step % self.val_interval == 0 and self.global_step > 0:
                val_results = self.validate()
                self.log_validation_results(val_results)
                
                # Save best model
                if val_results['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['total_loss']
                    self.save_checkpoint('best_model.pth')
            
            # Checkpointing
            if self.global_step % self.save_interval == 0 and self.global_step > 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')
            
            # Visualization
            if self.global_step % self.vis_interval == 0 and self.global_step > 0:
                self.visualize_predictions(batch)
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        epoch_avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return {**epoch_avg_losses, **epoch_avg_metrics}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Sample random time steps for flow matching
        batch_size = batch['start_poses'].shape[0]
        time_steps = torch.rand(batch_size, 1, device=self.device)
        
        # Forward pass with mixed precision
        if self.use_amp:
            with autocast():
                # Note: We need to handle the loss computation specially
                # since our loss function needs access to the model
                loss, loss_details = self.criterion(
                    predicted_velocity=None,  # Will be computed inside loss
                    start_poses=batch['start_poses'],
                    goal_poses=batch['goal_poses'],
                    time_steps=time_steps,
                    point_clouds=batch['point_clouds'],
                    geometries=batch['geometries'],
                    model=self.model,
                    return_details=True
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, loss_details = self.criterion(
                predicted_velocity=None,
                start_poses=batch['start_poses'],
                goal_poses=batch['goal_poses'],
                time_steps=time_steps,
                point_clouds=batch['point_clouds'],
                geometries=batch['geometries'],
                model=self.model,
                return_details=True
            )
            
            loss.backward()
            self.optimizer.step()
        
        # Compute additional metrics
        # Get predicted velocity for metrics computation
        with torch.no_grad():
            interpolated_poses = self.model.se3_utils.geodesic_interpolation(
                batch['start_poses'], batch['goal_poses'], time_steps.squeeze(-1)
            )
            predicted_velocity = self.model(
                interpolated_poses, batch['goal_poses'], 
                batch['point_clouds'], batch['geometries'], time_steps
            )
            target_velocity = self.model.se3_utils.geodesic_velocity(
                batch['start_poses'], batch['goal_poses'], time_steps.squeeze(-1)
            )
            
            velocity_metrics = compute_velocity_metrics(predicted_velocity, target_velocity)
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss_details, velocity_metrics
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        val_losses = {}
        val_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Sample time steps
                batch_size = batch['start_poses'].shape[0]
                time_steps = torch.rand(batch_size, 1, device=self.device)
                
                # Forward pass
                loss, loss_details = self.criterion(
                    predicted_velocity=None,
                    start_poses=batch['start_poses'],
                    goal_poses=batch['goal_poses'],
                    time_steps=time_steps,
                    point_clouds=batch['point_clouds'],
                    geometries=batch['geometries'],
                    model=self.model,
                    return_details=True
                )
                
                # Accumulate validation losses
                for key, value in loss_details.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item() if torch.is_tensor(value) else value)
        
        self.model.train()
        
        # Compute averages
        val_avg_losses = {f'val_{k}': np.mean(v) for k, v in val_losses.items()}
        
        return val_avg_losses
    
    def visualize_predictions(self, batch: Dict[str, torch.Tensor]):
        """Visualize model predictions"""
        if self.rank != 0:
            return
        
        self.model.eval()
        
        with torch.no_grad():
            # Take first sample from batch
            sample_idx = 0
            start_pose = batch['start_poses'][sample_idx:sample_idx+1]
            goal_pose = batch['goal_poses'][sample_idx:sample_idx+1]
            point_cloud = batch['point_clouds'][sample_idx:sample_idx+1]
            geometry = batch['geometries'][sample_idx:sample_idx+1]
            
            # Generate trajectory
            trajectory, times = self.model.generate_trajectory(
                start_pose.squeeze(0), goal_pose.squeeze(0),
                point_cloud.squeeze(0), geometry.squeeze(0)
            )
            
            # Create visualization
            vis_path = self.vis_dir / f'trajectory_step_{self.global_step}.png'
            self.evaluator.visualize_trajectory(
                trajectory, point_cloud.squeeze(0), str(vis_path)
            )
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'trajectory_visualization': wandb.Image(str(vis_path)),
                    'nfe_count': self.model.get_nfe()
                }, step=self.global_step)
        
        self.model.train()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        self.log(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Load scaler
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.log(f"Checkpoint loaded: {checkpoint_path}")
    
    def log_training_step(self, batch_idx: int, loss_dict: Dict, metrics_dict: Dict):
        """Log training step"""
        if self.rank != 0:
            return
        
        # Console and file logging
        log_msg = f"Epoch {self.current_epoch}, Step {self.global_step}, Batch {batch_idx}"
        log_msg += f" | Loss: {loss_dict.get('total_loss', 0.0):.6f}"
        log_msg += f" | FM: {loss_dict.get('fm_loss', 0.0):.6f}"
        log_msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        self.log(log_msg)
        
        # Wandb logging
        if self.use_wandb:
            log_data = {
                'train/total_loss': loss_dict.get('total_loss', 0.0),
                'train/fm_loss': loss_dict.get('fm_loss', 0.0),
                'train/collision_loss': loss_dict.get('collision_loss', 0.0),
                'train/regularization_loss': loss_dict.get('regularization_loss', 0.0),
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/epoch': self.current_epoch,
                **{f'train/{k}': v for k, v in metrics_dict.items()}
            }
            wandb.log(log_data, step=self.global_step)
    
    def log_validation_results(self, val_results: Dict[str, float]):
        """Log validation results"""
        if self.rank != 0:
            return
        
        log_msg = f"Validation | Total Loss: {val_results.get('val_total_loss', 0.0):.6f}"
        self.log(log_msg)
        
        if self.use_wandb:
            wandb.log(val_results, step=self.global_step)
    
    def log(self, message: str):
        """Log message"""
        if self.rank == 0:
            print(f"[Rank {self.rank}] {message}")
            logging.info(message)
    
    def train(self):
        """Main training loop"""
        self.log(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            epoch_results = self.train_epoch()
            
            # Log epoch results
            if self.rank == 0:
                epoch_msg = f"Epoch {epoch} complete | "
                epoch_msg += f"Avg Loss: {epoch_results.get('total_loss', 0.0):.6f}"
                self.log(epoch_msg)
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        **{f'epoch_{k}': v for k, v in epoch_results.items()}
                    }, step=self.global_step)
        
        self.log("Training complete!")
        
        # Final checkpoint
        if self.rank == 0:
            self.save_checkpoint('final_model.pth')


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_process(rank: int, world_size: int, config: Dict[str, Any]):
    """Training process for distributed training"""
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Setup random seeds
    seed = config.get('seed', 42) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create trainer and train
    trainer = SE3RFMTrainer(config, rank, world_size)
    
    # Load checkpoint if specified
    if config.get('resume_from'):
        trainer.load_checkpoint(config['resume_from'])
    
    trainer.train()
    
    if world_size > 1:
        cleanup_distributed()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='SE(3) RFM Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    
    # Add resume path if specified
    if args.resume:
        config['resume_from'] = args.resume
    
    # Multi-GPU training
    world_size = args.num_gpus if torch.cuda.is_available() else 1
    
    if world_size > 1:
        mp.spawn(
            train_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        train_process(0, 1, config)


if __name__ == '__main__':
    main()