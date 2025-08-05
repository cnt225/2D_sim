"""
Learning Rate Schedulers for SE(3) RFM Training

Provides various learning rate scheduling strategies for training
SE(3) Riemannian Flow Matching models.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any
import math


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine_annealing',
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler by type
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments
        
    Returns:
        scheduler: Configured learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 100),
            gamma=kwargs.get('gamma', 0.5)
        )
    
    elif scheduler_type == 'multi_step':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [100, 200, 300]),
            gamma=kwargs.get('gamma', 0.5)
        )
    
    elif scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.99)
        )
    
    elif scheduler_type == 'cosine_annealing':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 1000),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'cosine_annealing_warm_restarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 100),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'linear_warmup_cosine':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 1000),
            max_steps=kwargs.get('max_steps', 10000),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class LinearWarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing learning rate scheduler
    
    Performs linear warmup for specified steps, then cosine annealing decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_steps = self.last_epoch - self.warmup_steps
            cosine_max_steps = self.max_steps - self.warmup_steps
            
            if cosine_max_steps <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_steps / cosine_max_steps))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]


class CyclicCosineAnnealingLR(lr_scheduler._LRScheduler):
    """
    Cyclic cosine annealing learning rate scheduler
    
    Performs multiple cycles of cosine annealing with optional decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cycle_length: int = 1000,
        num_cycles: int = 5,
        eta_min: float = 1e-6,
        cycle_decay: float = 1.0,
        last_epoch: int = -1
    ):
        self.cycle_length = cycle_length
        self.num_cycles = num_cycles
        self.eta_min = eta_min
        self.cycle_decay = cycle_decay
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_cycle = min(self.last_epoch // self.cycle_length, self.num_cycles - 1)
        cycle_position = self.last_epoch % self.cycle_length
        
        # Cosine annealing within cycle
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_position / self.cycle_length))
        
        # Apply cycle decay
        cycle_decay_factor = self.cycle_decay ** current_cycle
        
        return [
            self.eta_min + (base_lr * cycle_decay_factor - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


class PolynomialLR(lr_scheduler._LRScheduler):
    """
    Polynomial learning rate decay scheduler
    
    Decays learning rate using polynomial function.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int = 10000,
        power: float = 0.9,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.max_steps = max_steps
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.max_steps:
            return [self.eta_min for _ in self.base_lrs]
        
        decay_factor = (1 - self.last_epoch / self.max_steps) ** self.power
        
        return [
            self.eta_min + (base_lr - self.eta_min) * decay_factor
            for base_lr in self.base_lrs
        ]


def get_scheduler_with_config(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get scheduler from configuration dictionary
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        scheduler: Configured learning rate scheduler
    """
    scheduler_type = config.get('type', 'cosine_annealing')
    scheduler_args = {k: v for k, v in config.items() if k != 'type'}
    
    return get_scheduler(optimizer, scheduler_type, **scheduler_args)


class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on training metrics
    
    Adjusts learning rate based on loss plateau detection and other metrics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
        cooldown: int = 5
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.cooldown = cooldown
        
        self.best_loss = float('inf')
        self.wait_count = 0
        self.cooldown_count = 0
        self.last_lr_update = 0
    
    def step(self, current_loss: float, epoch: int):
        """
        Update learning rate based on current loss
        
        Args:
            current_loss: Current validation loss
            epoch: Current epoch number
        """
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return
        
        # Check if loss improved
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # Reduce learning rate if no improvement
        if self.wait_count >= self.patience:
            self._reduce_lr(epoch)
            self.wait_count = 0
            self.cooldown_count = self.cooldown
    
    def _reduce_lr(self, epoch: int):
        """Reduce learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                print(f"Epoch {epoch}: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                self.last_lr_update = epoch
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]