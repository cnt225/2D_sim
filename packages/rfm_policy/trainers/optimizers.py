"""
Optimizers for SE(3) RFM Training

Provides various optimizers with proper configurations for training
SE(3) Riemannian Flow Matching models.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Iterator
from torch.nn.parameter import Parameter


def get_optimizer(
    parameters: Iterator[Parameter],
    optimizer_type: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer by type with specified parameters
    
    Args:
        parameters: Model parameters to optimize
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        optimizer: Configured optimizer
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            momentum=kwargs.get('momentum', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class AdamWWithWarmup(torch.optim.Optimizer):
    """
    AdamW optimizer with learning rate warmup
    
    Combines AdamW with linear warmup for better training stability.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
        **kwargs
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            warmup_steps=warmup_steps
        )
        super().__init__(params, defaults)
        
        self.step_count = 0
    
    def step(self, closure=None):
        """Perform optimization step with warmup"""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        for group in self.param_groups:
            # Apply warmup
            warmup_factor = min(1.0, self.step_count / group['warmup_steps'])
            current_lr = group['lr'] * warmup_factor
            
            # Standard AdamW update
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = current_lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * current_lr)
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


def get_optimizer_with_config(model_parameters, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Get optimizer from configuration dictionary
    
    Args:
        model_parameters: Model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        optimizer: Configured optimizer
    """
    opt_type = config.get('type', 'adamw')
    
    if opt_type == 'adamw_warmup':
        return AdamWWithWarmup(
            model_parameters,
            lr=config.get('lr', 0.001),
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_steps=config.get('warmup_steps', 1000)
        )
    else:
        return get_optimizer(
            model_parameters,
            optimizer_type=opt_type,
            **{k: v for k, v in config.items() if k != 'type'}
        )