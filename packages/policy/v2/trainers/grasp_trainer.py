import os
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import logging
import torch
from copy import deepcopy
from torch.distributed import destroy_process_group

from losses import get_loss
from utils.metrics import MMDCalculator, EMDCalculator, averageMeter
from trainers.schedulers import get_scheduler
from utils.mesh import scene_generator, meshes_to_numpy


num_grasps = 100


class GraspTrainer:
    """Trainer for a training of grasp sampler model"""
    def __init__(self, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.d_val_result = {}
        self.best_model_criteria = self.cfg.get('best_model_criteria', None)
        self.ddp = self.cfg.get('ddp', False)
        self.global_rank = int(os.environ['RANK']) if self.ddp else 0
        self.losses = [get_loss(cfg_loss) for cfg_loss in training_cfg['losses'].values()] if hasattr(training_cfg, 'losses') else None
        self.mmd_calculator = MMDCalculator(type='SE3', bandwidth_base=5)
        self.emd_calculator = EMDCalculator(type='SE3')

    def train(self, model, optimizer, d_dataloaders, logger=None, logdir=''):
        if self.ddp:
            model = DDP(model, device_ids=[self.device])

        cfg = self.cfg

        if not hasattr(cfg, 'early_stopping_criterian'):
            cfg['early_stopping_criterian'] = np.inf

        if not hasattr(cfg, 'save_epochs'):
            cfg['save_epoch'] = np.inf

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

            for data in train_loader:
                i += 1

                model.train()

                for key, val in data.items():
                    data[key] = val.to(self.device)

                tic = time.time()

                d_train = model.module.train_step(data, self.losses, optimizer) if self.ddp else model.train_step(data, self.losses, optimizer)

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

                    for val_data in val_loader:
                        for key, val in val_data.items():
                            val_data[key] = val.to(self.device)

                        d_val = model.module.val_step(val_data, self.losses) if self.ddp else model.val_step(val_data, self.losses)

                        logger.process_iter_val(d_val)

                    toc = time.time()

                    d_val = logger.summary_val(i)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for validation: {toc - tic:.4f}")
                    print(d_val['print_str'])
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for validation: {toc - tic:.4f}")
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

                    num_grasps_split = val_loader.dataset.num_grasps if hasattr(val_loader.dataset, 'num_grasps') else None
                    scale = 1 / val_loader.dataset.scale if hasattr(val_loader.dataset, 'scale') else 1

                    mmd_list_class = []
                    emd_list_class = []

                    for pc_list_object, Ts_grasp_list_object in zip(val_loader.dataset.pc_list_class, val_loader.dataset.Ts_grasp_list_class):
                        mmd_list_object = []
                        emd_list_object = []

                        for pc_list_rot, Ts_grasp_list_rot in zip(pc_list_object, Ts_grasp_list_object):
                            mmd_list_rot = []
                            emd_list_rot = []

                            pc_list_rot = torch.Tensor(np.stack(pc_list_rot)).to(self.device)
                            Ts_grasp_list_rot = [torch.Tensor(Ts_grasp).to(self.device) for Ts_grasp in Ts_grasp_list_rot]

                            Ts_grasp_pred_list = model.module.eval_step(pc_list_rot, num_grasps, num_grasps_split) if self.ddp else model.eval_step(pc_list_rot, num_grasps, num_grasps_split)

                            for Ts_grasp_pred, Ts_grasp_target in zip(Ts_grasp_pred_list, Ts_grasp_list_rot):
                                Ts_grasp_pred[:, :3, 3] *= scale
                                Ts_grasp_target[:, :3, 3] *= scale

                                mmd_list_rot += [self.mmd_calculator(Ts_grasp_pred, Ts_grasp_target)]
                                emd_list_rot += [self.emd_calculator(Ts_grasp_pred, Ts_grasp_target)]

                            mmd_list_object += [sum(mmd_list_rot) / len(mmd_list_rot)]
                            emd_list_object += [sum(emd_list_rot) / len(emd_list_rot)]

                        mmd_list_class += [sum(mmd_list_object) / len(mmd_list_object)]
                        emd_list_class += [sum(emd_list_object) / len(emd_list_object)]

                    toc = time.time()

                    d_eval = {'mmd_': sum(mmd_list_class) / len(mmd_list_class), 'emd_': sum(emd_list_class) / len(emd_list_class)}

                    logger.logging(i, d_eval)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for evaluation: {toc - tic:.4f}")
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for evaluation: {toc - tic:.4f}")

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

                    idxs_class = np.random.choice(len(val_loader.dataset.pc_list_class), size=3, replace=True)
                    idxs_object = [np.random.randint(len(val_loader.dataset.pc_list_class[idx_class])) for idx_class in idxs_class]
                    idxs_rot = [np.random.randint(len(val_loader.dataset.pc_list_class[idx_class][idx_object])) for idx_class, idx_object in zip(idxs_class, idxs_object)]

                    pc_list = torch.Tensor(np.array([val_loader.dataset.pc_list_class[idx_class][idx_object][idx_rot] for idx_class, idx_object, idx_rot in zip(idxs_class, idxs_object, idxs_rot)])).to(self.device)
                    mesh_list = deepcopy([val_loader.dataset.mesh_list_class[idx_class][idx_object][idx_rot] for idx_class, idx_object, idx_rot in zip(idxs_class, idxs_object, idxs_rot)])
                    Ts_grasp_target_list = [val_loader.dataset.Ts_grasp_list_class[idx_class][idx_object][idx_rot] for idx_class, idx_object, idx_rot in zip(idxs_class, idxs_object, idxs_rot)]
                    Ts_grasp_target_list = [Ts_grasp_target[np.random.choice(len(Ts_grasp_target), size=10, replace=False)] for Ts_grasp_target in Ts_grasp_target_list]

                    scale = 1 / val_loader.dataset.scale if hasattr(val_loader.dataset, 'scale') else 1

                    Ts_grasp_pred_list, img_dict = model.module.vis_step(pc_list) if self.ddp else model.vis_step(pc_list)

                    for mesh, Ts_grasp_pred, Ts_grasp_target in zip(mesh_list, Ts_grasp_pred_list, Ts_grasp_target_list):
                        mesh.scale(scale, center=[0, 0, 0])
                        Ts_grasp_pred[:, :3, 3] *= scale
                        Ts_grasp_target[:, :3, 3] *= scale

                    mesh_pred_list = scene_generator(mesh_list, Ts_grasp_pred_list)
                    mesh_target_list = scene_generator(mesh_list, Ts_grasp_target_list)

                    vertices_pred, faces_pred, colors_pred = meshes_to_numpy(mesh_pred_list)
                    vertices_target, faces_target, colors_target = meshes_to_numpy(mesh_target_list)

                    d_vis = {
                        'mesh_pred%': [vertices_pred, faces_pred, colors_pred], 
                        'mesh_target%': [vertices_target, faces_target, colors_target]
                    }
                    d_vis.update(img_dict)

                    toc = time.time()

                    logger.logging(i, d_vis)

                    print(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for visaulization: {toc - tic:.4f}")
                    logging.info(f"Epoch [{i_epoch:d}] Iter [{i:d}] Elapsed time for visaulization: {toc - tic:.4f}")

                if i % cfg['save_interval'] == 0 and self.global_rank == 0:
                    self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_iter=i)

                if break_count == cfg['early_stopping_criterian']:
                    print(f"Early Stopping!: (val_loss > best_val_loss) {break_count} times")
                    logging.info(f"Early Stopping!: (val_loss > best_val_loss) {break_count} times")

                    flag_break = True
                    break

            if i_epoch % cfg['save_epoch'] == 0:
                self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_epoch=i_epoch)

            if flag_break:
                break

        if self.global_rank == 0:
            self.save_model(model, optimizer, logdir, break_count, best_val_loss, best=False, i_iter='last')

        if self.ddp:
            destroy_process_group()

        return model, best_val_loss

    def save_model(self, model, opt, logdir, break_count, best_val_loss, best=False, i_iter=None, i_epoch=None, evalmodel=False):
        if best:
            pkl_name = 'model_best_eval.pkl' if evalmodel else 'model_best.pkl'
        else:
            pkl_name = f'model_iter_{i_iter}.pkl' if i_iter is not None else f'model_epoch_{i_epoch}.pkl'

        model_state = model.module.state_dict() if self.ddp else model.state_dict()

        state = {
            'epoch': i_epoch, 
            'model_state': model_state, 
            'iter': i_iter, 
            'optimizer': opt.state_dict(),
            'best_val_loss': best_val_loss,
            'break_count': break_count
        }

        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)

        print(f"Model saved: {pkl_name}")
