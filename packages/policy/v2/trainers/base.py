import os
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import logging
import torch
from torch.distributed import destroy_process_group

from utils.metrics import averageMeter
from trainers.schedulers import get_scheduler


class BaseTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.best_model_criteria = self.cfg.get('best_model_criteria', None)
        self.ddp = self.cfg.get('ddp', False)
        if self.ddp:
            self.global_rank = int(os.environ['RANK'])
        else:
            self.global_rank = 0

    def train(self, model, optimizer, d_dataloaders, logger=None, logdir=''):
        if self.ddp:
            model = DDP(model, device_ids=[self.device])

        cfg = self.cfg

        if not hasattr(cfg, 'early_stopping_criterian'):
            cfg['early_stopping_criterian'] = np.inf

        if not hasattr(cfg, 'save_epochs'):
            cfg['save_epochs'] = 100

        time_meter = averageMeter()

        train_loader, val_loader = d_dataloaders['train'], d_dataloaders['valid']

        best_val_loss = np.inf
        if self.best_model_criteria is not None:
            best_eval_loss = np.inf
        break_count = 0

        sch_config = self.cfg.get('scheduler', None)
        if sch_config is not None:
            lr_scheduler = get_scheduler(sch_config, optimizer)

        i = 0
        flag_break = False

        for i_epoch in range(1, cfg['n_epoch'] + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(i_epoch - 1)

            for x, y in train_loader:
                i += 1

                model.train()

                x = x.to(self.device)
                y = y.to(self.device)

                tic = time.time()

                if self.ddp:
                    d_train = model.module.train_step(x, y, optimizer)
                else:
                    d_train = model.train_step(x, y, optimizer)

                toc = time.time()
                time_meter.update(toc - tic)

                logger.process_iter_train(d_train)

                if sch_config is not None:
                    lr_scheduler.step()

                if i % cfg['print_interval'] == 0:
                    d_train = logger.summary_train(i)

                    print(f"GPU[{self.global_rank}] Epoch [{i_epoch:d}] Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    logging.info(f"GPU[{self.global_rank}] Epoch [{i_epoch:d}] Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")

                    time_meter.reset()

                if i % cfg['val_interval'] == 0:
                    # logger.val_loss_meter.reset()

                    model.eval()

                    tic = time.time()

                    for val_x, val_y in val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        if self.ddp:
                            d_val = model.module.val_step(val_x, val_y)
                        else:
                            d_val = model.val_step(val_x, val_y)

                        logger.process_iter_val(d_val)

                    toc = time.time()

                    d_val = logger.summary_val(i)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for validation: {toc-tic:.4f}")
                    print(d_val['print_str'])
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for validation: {toc-tic:.4f}")
                    logging.info(d_val['print_str'])

                    val_loss = d_val['loss/val_loss_']

                    if val_loss > best_val_loss:
                        break_count += 1
                    elif val_loss < best_val_loss and break_count > 0:
                        break_count -= 1

                    best_model = val_loss < best_val_loss

                    if best_model:
                        if self.global_rank == 0:
                            self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=best_model, i_iter=i)

                            print(f"Epoch [{i_epoch:d}] Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}")
                            logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}")

                        best_val_loss = val_loss

                if i % cfg['eval_interval'] == 0:
                    model.eval()

                    tic = time.time()

                    if self.cfg['eval_config']['exp_type'] == 'image':
                        val_xs = self.device
                    elif self.cfg['eval_config']['exp_type'] in ['2dtoy', '2dtoy_sphere']:
                        val_xs = []
                        for val_x, _ in val_loader:
                            val_xs.append(val_x.to(self.device))
                        val_xs = torch.cat(val_xs, dim=0)
                    elif self.cfg['eval_config']['exp_type'] == 'I2S':
                        val_xs = []
                        val_ys = []
                        for val_x, val_y in val_loader:
                            val_xs.append(val_x.to(self.device))
                            val_ys.append(val_y.to(self.device))
                        val_xs = torch.cat(val_xs, dim=0)
                        val_ys = torch.cat(val_ys, dim=0)

                    if self.ddp:
                        d_eval = model.module.eval_step(val_xs, **self.cfg['eval_config'])
                    else:
                        if self.cfg['eval_config']['exp_type'] == 'I2S':
                            d_eval = model.eval_step(val_xs, val_ys, **self.cfg['eval_config'])
                        else:
                            d_eval = model.eval_step(val_xs, **self.cfg['eval_config'])

                    toc = time.time()

                    logger.logging(i, d_eval)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for evaluation: {toc-tic:.4f}")
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for evaluation: {toc-tic:.4f}")

                    if self.global_rank == 0:
                        if self.best_model_criteria is not None:
                            eval_loss = d_eval[self.best_model_criteria]

                            best_eval_model = eval_loss < best_eval_loss

                            if best_eval_model:
                                print(f"Epoch [{i_epoch:d}] Iter [{i:d}] best eval model saved {eval_loss} <= {best_eval_loss}")
                                logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] best eval model saved {eval_loss} <= {best_eval_loss}")

                                best_eval_loss = eval_loss

                                self.save_model(model, optimizer, logdir, break_count, best_eval_loss, best=best_eval_model, i_iter=i, evalmodel=True)

                if i % cfg['vis_interval'] == 0:
                    model.eval()

                    tic = time.time()

                    if self.cfg['vis_config']['exp_type'] == '2dtoy' or self.cfg['vis_config']['exp_type'] == 'I2S':
                        xlim = train_loader.dataset.xlim
                        ylim = train_loader.dataset.ylim
                    elif self.cfg['vis_config']['exp_type'] == '2dtoy_sphere':
                        xlim = train_loader.dataset.xlim
                        ylim = train_loader.dataset.ylim
                        zlim = train_loader.dataset.zlim
                    else:
                        xlim = None
                        ylim = None

                    if self.ddp:
                        d_vis = model.module.vis_step(**self.cfg['vis_config'], device=self.device, xlim=xlim, ylim=ylim, train_dl=train_loader, val_dl=val_loader)
                    else:
                        if self.cfg['vis_config']['exp_type'] == '2dtoy_sphere':
                            d_vis = model.vis_step(**self.cfg['vis_config'], device=self.device, xlim=xlim, ylim=ylim, zlim=zlim, train_dl=train_loader, val_dl=val_loader)
                        else:
                            d_vis = model.vis_step(**self.cfg['vis_config'], device=self.device, xlim=xlim, ylim=ylim, train_dl=train_loader, val_dl=val_loader)

                    toc = time.time()

                    logger.logging(i, d_vis)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for visaulization: {toc-tic:.4f}")
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for visaulization: {toc-tic:.4f}")

                if i % cfg['save_interval'] == 0 and self.global_rank == 0:
                    self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_iter=i)

                if break_count == cfg['early_stopping_criterian']:
                    print(f"Early Stopping!: (val_loss > best_val_loss) {break_count} times")
                    logging.info(f"Early Stopping!: (val_loss > best_val_loss) {break_count} times")

                    flag_break = True
                    break

            if (i_epoch + 1) % cfg['save_epochs'] == 0:
                self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_epoch=i_epoch)

            if flag_break:
                break

        if self.global_rank == 0:
            self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_iter='last')

        if self.ddp:
            destroy_process_group()

        return model, best_val_loss

    def save_model(self, model, opt, logdir, Break_count, best_val_loss, best=False, i_iter=None, i_epoch=None, evalmodel=False):
        if best:
            if evalmodel:
                pkl_name = 'model_best_eval.pkl'
            else:
                pkl_name = 'model_best.pkl'
        else:
            if i_iter is not None:
                pkl_name = 'model_iter_{}.pkl'.format(i_iter)
            else:
                pkl_name = 'model_epoch_{}.pkl'.format(i_epoch)

        if self.ddp:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            'epoch': i_epoch, 
            'model_state': model_state, 
            'iter': i_iter, 
            'optimizer': opt.state_dict(),
            'best_val_loss': best_val_loss,
            'Break_count': Break_count}

        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)

        print(f"Model saved: {pkl_name}")
