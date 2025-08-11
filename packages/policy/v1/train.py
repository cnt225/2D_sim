import argparse
from omegaconf import OmegaConf
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import logging
import wandb
import torch
import numpy as np
import random

from utils.utils import save_yaml
from utils.utils import ddp_setup
from loaders import get_dataloader
from models import get_model
from trainers import get_trainer, get_logger
from trainers.optimizers import get_optimizer


def run(cfg, writer):    
    # Setup seeds
    try:
        device_int = int(cfg['device'])
    except:
        device_int = 0

    seed = cfg.get('seed', 1) + device_int
    print(f"running with random seed : {seed}")
    logging.info(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    # Setup device
    device = cfg.device
    ddp = cfg['training'].get('ddp', False)
    if ddp:
        ddp_setup()
        global_rank = int(os.environ['RANK'])
    else:
        global_rank = 0

    # Setup Dataloader
    print(f"[GPU {global_rank}] Setup Dataloader ...")
    logging.info(f"[GPU {global_rank}] Setup Dataloader ...")
    d_dataloaders = {}
    for key, dataloader_cfg in cfg['data'].items():
        d_dataloaders[key] = get_dataloader(dataloader_cfg, ddp=ddp)

    # Setup Model
    print(f"[GPU {global_rank}] Setup Model ...")
    logging.info(f"[GPU {global_rank}] Setup Model ...")
    if cfg['model']['arch'] == 'fm' or cfg['model']['arch'] == 'rfm' or cfg['model']['arch'] == 'crfm' or cfg['model']['arch'] == 'grasp_rcfm' or cfg['model']['arch'] == 'motion_rcfm':
        kwargs = {'val_dl': d_dataloaders['valid'], 'device': device}
    else:
        kwargs = {}
    model = get_model(cfg['model'], **kwargs).to(device)

    trainer = get_trainer(cfg)

    logger = get_logger(cfg, writer, ddp=ddp)

    # Setup optimizer, lr_scheduler and loss function
    print(f"[GPU {global_rank}] Setup optimizer ...")
    logging.info(f"[GPU {global_rank}] Setup optimizer ...")
    if cfg['model'].get('prob_path', None) in ['GPP', 'MAPP', 'Manifold_PP']:
        optimizer = get_optimizer(cfg['training']['optimizer'], model.velocity_field.parameters())
    else:
        optimizer = get_optimizer(cfg['training']['optimizer'], model.parameters())

    # Training starts
    print(f"[GPU {global_rank}] Training starts ...")
    logging.info(f"[GPU {global_rank}] Training starts ...")
    model, _ = trainer.train(model, optimizer, d_dataloaders, logger=logger, logdir=writer.file_writer.get_logdir())

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except ValueError:
        return val

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}

    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]

        assert '=' not in key, "optional arguments should be separated by space"

        kwargs[key.strip('-')] = parse_arg_type(val)

    return kwargs


def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}

    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg

        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}

                d = d[each_key]

    return d_new_cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default='any')
    parser.add_argument('--logdir', default='train_results')
    parser.add_argument('--run', default=None)

    args, unknown = parser.parse_known_args()

    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)    
    print(d_cmd_cfg)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    if args.device == 'cpu':
        cfg['device'] = f'cpu'
        global_rank = 0
    elif args.device == 'any':
        cfg['device'] = 'cuda'
        global_rank = 0
    elif args.device == 'ddp':
        cfg['device'] = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
    else:
        cfg['device'] = f'cuda:{args.device}'
        global_rank = 0

    if args.run is None:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        run_id = args.run

    config_basename = os.path.basename(args.config).split('.')[0]

    if args.logdir is None:
        logdir = os.path.join(cfg['logdir'], config_basename, str(run_id))
    else:
        logdir = os.path.join(args.logdir, config_basename, str(run_id))

    print_ = ((global_rank == 0) and (args.device == 'ddp')) or (args.device != 'ddp')

    writer = SummaryWriter(logdir=logdir)
    logging.basicConfig(
        filename=os.path.join(logdir, 'logging.log'),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG
    )
    if print_:
        print("Result directory: {}".format(logdir))
        logging.info("Result directory: {}".format(logdir))

    # copy config file
    if print_:
        copied_yml = os.path.join(logdir, os.path.basename(args.config))
        save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
        print(f"config saved as {copied_yml}")
        logging.info(f"config saved as {copied_yml}")

    if cfg.get('wandb', False):
        if print_:
            wandb.init(entity=cfg['wandb']['entity'], project=cfg['wandb']['project_name'], config=OmegaConf.to_container(cfg), name=logdir)

    run(cfg, writer)
