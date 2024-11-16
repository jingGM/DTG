import argparse
import os

import torch
from src.utils.configs import TrainingConfig, GeneratorType, DiffusionModelType, CRNNType


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--name', type=str, default="hnav", help="name of project")
    parser.add_argument('--wandb_api', type=str, default="", help="Your wandb api")
    parser.add_argument('--only_load_model', action='store_true', default=False,
                        help='only load model to continue training')
    parser.add_argument('--snapshot', type=str, default="", help='snapshot')
    parser.add_argument('--evaluation_freq', type=int, default=5, help="evaluation frequency")
    parser.add_argument('--train_time_steps', type=int, default=32, help="time steps for training")
    parser.add_argument('--training_type', type=int, default=1, help="0: 100 epochs; 1: 30 epochs")
    parser.add_argument('--debug_output', type=str, default=None, help='snapshot')

    # data args:
    parser.add_argument('--data_root', type=str, help='root of the dataset', default="data_sample")
    parser.add_argument('--batch_size', type=int, default=2, help="the negative number in the same frame")
    parser.add_argument('--workers', type=int, default=16, help="the worker number in the dataloader")

    # model args:
    parser.add_argument('--generator_type', type=int, default=0, help="0: diffusion; 1: cvae")
    parser.add_argument('--diffusion_model', type=int, default=0, help="0: rnn; 1: unet")
    parser.add_argument('--crnn_type', type=int, default=0, help="0: gru; 1: lstm")
    parser.add_argument('--train_poses', action='store_true', default=False, help="if train poses or increments")
    parser.add_argument('--use_traversability', action='store_true', default=False, help="if train traversability")
    parser.add_argument('--traversable_steps', type=int, default=10, help="time steps used for traversability training")
    parser.add_argument('--diffusion_time_steps', type=int, default=100, help="")

    # GPUs
    parser.add_argument('--channels-last', action='store_true', default=False, help='Use channels_last memory layout')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-bn', action='store_true', default=False,
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--split-bn', action='store_true', help='Enable separate BN layers per augmentation split.')
    parser.add_argument('--device', type=int, default=-1, help="the gpu id")
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    return parser.parse_args()


def get_configuration():
    args = get_args()
    cfg = TrainingConfig

    #########################################
    # data configurations
    #########################################
    cfg.data.name = cfg.name = args.name
    cfg.data.root = cfg.loss.root = args.data_root
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.workers

    #########################################
    # model configurations
    #########################################
    if args.generator_type == 0:
        cfg.model.generator_type = cfg.loss.generator_type = GeneratorType.diffusion
    elif args.generator_type == 1:
        cfg.model.generator_type = cfg.loss.generator_type = GeneratorType.cvae

    else:
        raise Exception("decoder type is not defined")

    if args.diffusion_model == 0:
        cfg.model.diffusion.model_type = DiffusionModelType.crnn
        if args.crnn_type == 0:
            cfg.model.diffusion.crnn.type = CRNNType.gru
        else:
            cfg.model.diffusion.crnn.type = CRNNType.lstm
    elif args.diffusion_model == 1:
        cfg.model.diffusion.model_type = DiffusionModelType.unet
    else:
        raise Exception("diffusion model type is not defined")

    cfg.model.diffusion.num_train_timesteps = args.diffusion_time_steps
    cfg.model.diffusion.traversable_steps = args.traversable_steps
    cfg.loss.train_poses = args.train_poses
    if args.train_poses:
        cfg.loss.scale_waypoints = 20.0
    else:
        cfg.loss.scale_waypoints = 1.0

    #########################################
    # training configurations
    #########################################
    cfg.only_model = args.only_load_model
    cfg.snapshot = args.snapshot
    cfg.wandb_api = args.wandb_api
    cfg.evaluation_freq = args.evaluation_freq
    cfg.train_time_steps = args.train_time_steps
    cfg.loss.output_dir = args.debug_output

    if args.training_type == 0:
        cfg.max_epoch = 100
        cfg.lr_tm = 10
    elif args.training_type == 1:
        cfg.max_epoch = 30
        cfg.lr = 2e-5
        cfg.lr_tm = 30
        cfg.lr_min = 1e-8
    else:
        raise ValueError("the data type is not defined")

    # Devices
    if args.device >= 0:
        cfg.gpus.device = "cuda:{}".format(args.device)
    elif args.device == -1:
        cfg.gpus.device = "cuda"
        print("------------------- CUDA: ", torch.cuda.is_available(), "-----------------------")
    else:
        cfg.gpus.device = "cpu"
    # print(" ------ local rank: ", os.environ['LOCAL_RANK'])
    cfg.gpus.channels_last = args.channels_last
    cfg.gpus.local_rank = args.local_rank
    cfg.gpus.sync_bn = args.sync_bn
    cfg.gpus.split_bn = args.split_bn
    cfg.gpus.no_ddp_bb = args.no_ddp_bb

    return cfg