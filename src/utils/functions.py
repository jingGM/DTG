import random

import numpy as np
from typing import Union, Optional, Tuple, List
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


def get_device(device: Union[torch.device, str] = "cuda") -> torch.device:
    """
    get the device of the input string or torch.device
    Args:
        device: string or torch device

    Returns:
        the device format that can be used for torch tensors
    """
    if isinstance(device, str):
        assert device == "cuda" or device == "cuda:0" or device == "cuda:1" or device == "cpu", \
            "device should only be 'cuda' or 'cpu' "
    device = torch.device(device)
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def to_device(x, device):
    """
    put the input to a torch tensor in the given device
    Args:
        x: list, tuple of torch tensors, dict of torch tensors or torch tensors
        device: pytorch device
    Returns:
        the input data in the given device
    """
    if isinstance(x, list):
        x = [to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        x = (to_device(item, device) for item in x)
    elif isinstance(x, dict):
        x = {key: to_device(value, device) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if device == "cuda":
            x = x.cuda()
        else:
            x = x.to(device)
    return x


def release_cuda(x):
    """
    put the torch tensors from cuda to numpy
    Args:
        x: in put torch tensor, list of, dict of or tuple of torch tensors in cuda

    Returns:
        input data in local numpy
    """
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def inverse_transform(pts, transformation):
    if len(pts.shape) == 1:
        pts = pts[None, :]
    elif len(pts.shape) == 2:
        pass
    else:
        raise Exception("points shape is not correct")
    if pts.shape[-1] == 2:
        pts = np.concatenate((pts, np.ones_like(pts[:, :1]) * transformation[2, -1]), axis=-1)
    if transformation.shape[0] == 3:
        last_vector = np.zeros(4)[None, :]
        last_vector[0, -1] = 1
        transformation = np.concatenate((transformation, last_vector), axis=0)
    inv_transformation = np.linalg.inv(transformation)
    new_pts = appy_tranformation(pts, inv_transformation)
    return new_pts


def appy_tranformation(points, transform):
    assert transform.shape[1] == 4 and transform.shape[0] >= 3, "transform shape should be (3,4) or (4,4)"
    assert points.shape[1] == 3, "points shape should be (n,3)"
    if len(points.shape) == 2:
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        points = np.matmul(points, rotation.transpose(-1, -2)) + translation
    elif len(points.shape) == 3:
        rotation = transform[:3, :3]  # (3, 3)
        translation = transform[None, :3, 3]  # (1, 3)
        positions = np.matmul(points[:, :, -1], rotation.transpose(-1, -2)) + translation
        orientations = np.moveaxis(np.matmul(np.moveaxis(points[:, :, :-1], -2, -1),
                                             rotation.transpose(-1, -2)), -1, -2)
        points = np.concatenate((orientations, np.expand_dims(positions, axis=-1)), axis=-1)
    else:
        raise Exception("points shape is not correct: Nx3 or Nx3x4")
    return points