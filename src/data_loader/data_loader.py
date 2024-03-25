import copy
import os
import pickle
import random
import warnings
import cv2
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.data_loader.dataset import TrainData


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def registration_collate_fn_stack_mode(data_dicts):
    r"""Collate function for registration in stack mode.
    Args:
        data_dicts (List[Dict])
    Returns:
        collated_dict (Dict)
    """
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            value = torch.from_numpy(np.asarray(value)).to(torch.float)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)
    for key, value in collated_dict.items():
        collated_dict[key] = torch.stack(value, dim=0)
    return collated_dict


def get_data_loader(cfg, train=True):
    dataset = TrainData(cfg=cfg, train=train)
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        sampler=sampler,
        collate_fn=partial(registration_collate_fn_stack_mode),
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader


def train_data_loader(cfg):
    """
    This function is to create a training dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfgs, train=True)


def evaluation_data_loader(cfg):
    """
    This function is to create a evaluation dataloader with pytorch interface
    Args:
        cfg: The configuration of the dataset
    Returns:
        a dataloader in pytorch format
    """
    cfgs = copy.deepcopy(cfg)
    return get_data_loader(cfg=cfgs, train=False)