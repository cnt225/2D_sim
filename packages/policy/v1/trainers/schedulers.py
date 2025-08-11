import logging
from torch.optim.lr_scheduler import _LRScheduler

# from torch.optim.lr_scheduler import PolynomialLR

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, warm_up_steps=45000, max_decay_steps=2100000, end_learning_rate=1.0e-8, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.warm_up_steps = warm_up_steps
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        
        if self.last_step <= self.warm_up_steps:
            warmup_lrs = [(base_lr - self.end_learning_rate) * 
                            ((self.last_step / self.warm_up_steps) ** (self.power)) + 
                            self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group['lr'] = lr
                
        elif self.last_step <= self.max_decay_steps:
            decay_lrs = [-(base_lr - self.end_learning_rate) * 
                            (((self.last_step - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)) ** (self.power)) + 
                            base_lr for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
                

logger = logging.getLogger("ptsemseg")

key2opt = {
    "polynomial": PolynomialLRDecay
}


def get_scheduler(sch_dict, optimizer):
    scheduler = _get_scheduler_instance(sch_dict)

    params = {k: v for k, v in sch_dict.items() if k != "name"}

    scheduler = scheduler(optimizer, **params)
    return scheduler


def _get_scheduler_instance(sch_dict):
    sch_name = sch_dict["name"]
    if sch_name not in key2opt:
        raise NotImplementedError("scheduler {} not implemented".format(sch_name))

    logger.info("Using {} scheduler".format(sch_name))
    return key2opt[sch_name]