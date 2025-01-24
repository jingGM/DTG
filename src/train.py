import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess

from warnings import warn
import torch
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import datetime, timedelta

from src.utils.configs import TrainingConfig, ScheduleMethods, LossNames, LogNames, LogTypes
from src.loss import Loss
from src.models.model import get_model
from src.utils.functions import to_device, get_device, release_cuda
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader


class Trainer:
    def __init__(self, cfgs: TrainingConfig):
        """
        This class is the trainner
        Args:
            cfgs: the configuration of the training class
        """
        self.name = cfgs.name
        self.max_epoch = cfgs.max_epoch
        self.evaluation_freq = cfgs.evaluation_freq
        self.train_time_steps = cfgs.train_time_steps

        self.iteration = 0
        self.epoch = 0
        self.training = False

        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        if 'WORLD_SIZE' in os.environ and cfgs.gpus.device == "cuda":
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data.distributed = int(os.environ['WORLD_SIZE']) >= 1
            # log_name = self.name + "-" + str(int(os.environ['WORLD_SIZE'])) + "-" + str(
            #     int(os.environ['LOCAL_RANK'])) + "/" + datetime.now().strftime("%m-%d-%Y-%H-%M")
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data.distributed = False
            # log_name = self.name + "-" + datetime.now().strftime("%m-%d-%Y-%H-%M")

        # model
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        if self.device == torch.device("cpu"):
            pass
        else:
            self._set_model_gpus(cfgs.gpus)

        # set up loggers
        self.output_dir = cfgs.output_dir
        configs = {
            "lr": cfgs.lr,
            "lr_t0": cfgs.lr_t0,
            "lr_tm": cfgs.lr_tm,
            "lr_min": cfgs.lr_min,
            "gpus": cfgs.gpus,
            "epochs": self.max_epoch
        }
        wandb.login(key=cfgs.wandb_api)
        if self.distributed:
            self.wandb_run = wandb.init(project=self.name, config=configs, group="DDP")
        else:
            self.wandb_run = wandb.init(project=self.name, config=configs)

        # loss, optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay)
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, eta_min=cfgs.lr_min,
                                                                                  T_0=cfgs.lr_t0, T_mult=cfgs.lr_tm)
        else:
            raise ValueError("the current scheduler is not defined")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(state_dict)

        # loss functions
        if self.device == "cuda":
            self.loss_func = Loss(cfg=cfgs.loss).cuda()
        else:
            self.loss_func = Loss(cfg=cfgs.loss).to(self.device)

        # datasets:
        self.training_data_loader = train_data_loader(cfg=cfgs.data)
        self.evaluation_data_loader = evaluation_data_loader(cfg=cfgs.data)

        self.use_traversability = cfgs.loss.use_traversability
        self.generator_type = cfgs.model.generator_type
        self.time_step_loss_buffer = []
        self.time_step_number = cfgs.model.diffusion.traversable_steps
        self.traversability_threshold = cfgs.traversability_threshold

    def _set_model_gpus(self, cfg):
        # self.current_rank = 0  # global rank
        # cfg.local_rank = os.environ['LOCAL_RANK']
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            # this will make all .cuda() calls work properly
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            # self.logger.info\
            print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                  % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logger.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logger.info(
        #         f'Model created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # move model to GPU, enable channels last layout if set
        if self.distributed:
            self.model.cuda()
        else:
            self.model.to(self.device)

        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                print(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # setup distributed training
        if self.distributed:
            if cfg.local_rank == 0:
                print("Using native Torch DistributedDataParallel.")
            self.model = DDP(self.model, device_ids=[cfg.local_rank],
                             broadcast_buffers=not cfg.no_ddp_bb,
                             find_unused_parameters=True)
            # NOTE: EMA model does not need to be wrapped by DDP

        # # setup exponential moving average of model weights, SWA could be used here too
        # model_ema = None
        # if args.model_ema:
        #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        #     model_ema = ModelEmaV2(
        #         self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        print('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            warn('Missing keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            warn('Unexpected keys: {}'.format(unexpected_keys))
        print('Model has been loaded.')
        return state_dict

    def load_learning_parameters(self, state_dict):
        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            print('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            print('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                print('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                print('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def save_snapshot(self, filename):
        """
        save the snapshot of the model and other training parameters
        Args:
            filename: the output filename that is the full directory
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)
        # print('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(self.output_dir, str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        # print('Snapshot saved to "{}"'.format(snapshot_filename))

    def cleanup(self):
        dist.destroy_process_group()
        self.wandb_run.finish()

    def set_train_mode(self):
        """
        set the model to the training mode: parameters are differentiable
        """
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def optimizer_step(self):
        """
        run one step of the optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, data_dict, train=True) -> Tuple[dict, dict]:
        """
        one step of the model, loss function and also the metrics
        Args:
            data_dict: the input data dictionary
        Returns:
            the output from the model, the output from the loss function
        """
        # start_time = time.time()
        data_dict = to_device(data_dict, device=self.device)
        if train:
            output_dict = self.model(data_dict, sample=False)
            torch.cuda.empty_cache()
            loss_dict = self.loss_func(output_dict)
            output_dict.update(loss_dict)
            torch.cuda.empty_cache()
        else:
            output_dict = self.model(data_dict, sample=True)
            torch.cuda.empty_cache()
            eval_dict = self.loss_func.evaluate(output_dict)
            output_dict.update(eval_dict)
        return output_dict

    def update_log(self, results, timestep=None, log_name=None):
        if timestep is not None:
            self.wandb_run.log({LogNames.step_time: timestep})
        if log_name == LogTypes.train:
            value = self.scheduler.get_last_lr()
            self.wandb_run.log({log_name + "/" + LogNames.lr: value[-1]})

        if log_name is None:
            for key, value in results.items():
                self.wandb_run.log({key: value})
        else:
            for key, value in results.items():
                self.wandb_run.log({log_name + "/" + key: value})

    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        # with open(self.output_file, "a") as f:
        #     print("Training CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            self.iteration += 1
            data_dict[DataDict.traversable_step] = self.time_step_number
            for step_iteration in range(self.train_time_steps):
                output_dict = self.step(data_dict=data_dict)
                torch.cuda.empty_cache()

                output_dict[LossNames.loss].backward()
                self.optimizer_step()
                optimize_time = time.time()

                output_dict = release_cuda(output_dict)
                self.update_log(results=output_dict, timestep=optimize_time - last_time, log_name=LogTypes.train)
                last_time = time.time()
        self.scheduler.step()

        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth'.format(self.output_dir, self.name, self.epoch))

    def inference_epoch(self):
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):
        # if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0):
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                # if iteration % self.max_evaluation_iteration_per_epoch == 0 and iteration != 0:
                #     break
                start_time = time.time()
                output_dict = self.step(data_dict, train=False)
                torch.cuda.synchronize()
                step_time = time.time()
                output_dict = release_cuda(output_dict)
                torch.cuda.empty_cache()
                self.update_log(results=output_dict, timestep=step_time - start_time, log_name=LogTypes.others)

    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        for self.epoch in range(self.epoch, self.max_epoch, 1):
            self.set_eval_mode()
            self.inference_epoch()

            self.set_train_mode()
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
        self.cleanup()
