import copy
import math
import os
import pickle
import shutil
from os.path import join, exists

import cv2
import imageio
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from src.models.diff_hausdorf import HausdorffLoss
from src.utils.configs import GeneratorType, DataDict, Hausdorff, LossNames, DiffusionTypes


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()

        with open(join(cfg.root, "data.pkl"), "rb") as input_file:
            data = pickle.load(input_file)
        self.all_positions = data[DataDict.all_positions]
        self.network = data[DataDict.network]

        self.generator_type = cfg.generator_type
        self.use_traversability = cfg.use_traversability
        self.collision_distance = 0.5

        self.target_dis = nn.MSELoss(reduction="mean")
        self.distance = HausdorffLoss(mode=cfg.distance_type)

        self.train_poses = cfg.train_poses
        self.diffusion_type = cfg.diffusion_type
        self.distance_type = cfg.distance_type
        self.scale_waypoints = cfg.scale_waypoints
        self.last_ratio = cfg.last_ratio
        self.distance_ratio = cfg.distance_ratio
        self.vae_kld_ratio = cfg.vae_kld_ratio
        self.traversability_ratio = cfg.traversability_ratio

        self.map_resolution = cfg.map_resolution
        self.map_range = cfg.map_range
        self.output_dir = cfg.output_dir
        if self.output_dir:
            if not exists(self.output_dir):
                os.makedirs(self.output_dir)

    def _cropped_distance(self, path, single_map):
        N, Cp = path.shape
        M, Cs = single_map.shape
        assert Cs == Cp, "dimension should be the same, but get {}, {}".format(Cs, Cp)
        single_map = single_map.view(M, 1, Cs).to(torch.float)  # Mx1xC
        path = path.view(1, N, Cs)  # 1xNxC
        d = torch.min(torch.norm(single_map - path, dim=-1), dim=0)[0] * self.map_resolution  # N

        traversability = torch.clamp(d, 0.0001, self.collision_distance)
        values = traversability[torch.where(traversability < self.collision_distance)]
        if len(values) < 1:
            return torch.tensor(0, device=traversability.device), torch.tensor(1, device=traversability.device)
        else:
            torch.cuda.empty_cache()
            loss = torch.arctanh((self.collision_distance - values) / self.collision_distance)
            return loss.mean(), values.mean()

    def _local_collision(self, yhat, local_map):
        assert len(yhat.shape) == 3, "the shape should be B,N,2"
        # if len(yhat.shape) == 3:
        #     B, N, C = yhat.shape
        #     yhat = yhat.view(B, 1, N, C)
        By, N, C = yhat.shape
        Bl, W, H = local_map.shape
        assert Bl == By, "the batch shape {} and {} should be the same".format(By, Bl)
        assert W == H, "the local map width {} not equals to height {}".format(W, H)
        pixel_yhat = yhat / self.map_resolution + self.map_range
        pixel_yhat = pixel_yhat.to(torch.int)
        all_losses = []
        traversability_values = []
        for i in range(By):
            map_indices = torch.stack(torch.where(local_map[i] > 0), dim=1)
            loss, traversability = self._cropped_distance(pixel_yhat[i], map_indices)
            all_losses.append(loss)
            traversability_values.append(traversability)
        return torch.stack(all_losses), torch.stack(traversability_values)

    def forward_cvae(self, input_dict):
        mu = input_dict[DataDict.zmu]
        logvar = input_dict[DataDict.zvar]
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]
        y_last = ygt[:, -1, :]

        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        path_dis = self.distance(ygt, y_hat_poses).mean()
        last_pose_dis = self.target_dis(y_last, y_hat_poses[:, -1, :])
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y_last.shape[0]
        all_loss = self.distance_ratio * path_dis + self.last_ratio * last_pose_dis + self.vae_kld_ratio * kld_loss
        output = {
            LossNames.kld: kld_loss,
            LossNames.last_dis: last_pose_dis,
            LossNames.path_dis: path_dis,
        }

        if self.use_traversability:
            local_map = input_dict[DataDict.local_map]
            traversability_loss, traversability_values = self._local_collision(yhat=y_hat_poses, local_map=local_map)
            traversability_loss_mean = traversability_loss.mean()
            all_loss += self.traversability_ratio * traversability_loss_mean
            output.update({LossNames.traversability: traversability_loss_mean})

        output.update({LossNames.loss: all_loss})
        return output

    def forward_diffusion(self, input_dict):
        noise = input_dict[DataDict.noise]
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]

        output = {}
        if self.diffusion_type == DiffusionTypes.noise:
            all_loss = self.target_dis(y_hat, noise)
            if self.use_traversability:
                traversablility_hat = input_dict[DataDict.predict_path]
                if self.train_poses:
                    traversability_hat_poses = traversablility_hat * self.scale_waypoints
                else:
                    traversability_hat_poses = torch.cumsum(traversablility_hat, dim=1) * self.scale_waypoints
        elif self.diffusion_type == DiffusionTypes.trajectory:
            if self.train_poses:
                y_hat_poses = y_hat * self.scale_waypoints
            else:
                y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints
            if self.use_traversability:
                B, _, _ = y_hat.shape
                traversability_hat_poses = y_hat_poses[int(B / 2):]
                y_hat_poses = y_hat_poses[:int(B / 2)]

            path_dis = self.distance(ygt, y_hat_poses).mean()
            last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
            all_loss = self.distance_ratio * path_dis + self.last_ratio * last_pose_dis
            output.update({
                LossNames.last_dis: last_pose_dis,
                LossNames.path_dis: path_dis,
            })
        else:
            raise Exception("the diffusion type is not defined")

        if self.use_traversability:
            local_map = input_dict[DataDict.local_map]
            traversability_loss, traversability_values = self._local_collision(yhat=traversability_hat_poses, local_map=local_map)
            traversability_loss_mean = traversability_loss.to(float).mean()
            all_loss += self.traversability_ratio * traversability_loss_mean
            output.update({LossNames.traversability: traversability_loss_mean})

        output.update({LossNames.loss: all_loss})
        return output

    def forward_estimation(self, input_dict):
        gt = input_dict[DataDict.traversability_gt]
        est_loss, est_dict = self.forward_traversability(loss=gt,
                                                         mu=input_dict[DataDict.traversability_mu],
                                                         var=input_dict[DataDict.traversability_var],
                                                         val=input_dict[DataDict.traversability_pred])
        est_dict.update({LossNames.loss: est_loss})
        return est_dict

    def forward(self, input_dict):
        if self.generator_type == GeneratorType.cvae:
            return self.forward_cvae(input_dict=input_dict)
        elif self.generator_type == GeneratorType.diffusion:
            return self.forward_diffusion(input_dict=input_dict)
        elif self.generator_type == GeneratorType.estimator:
            return self.forward_estimation(input_dict=input_dict)

    def convert_path_pixel(self, trajectory):
        return np.clip(np.around(trajectory / self.map_resolution)[:, :2] + self.map_range, 0, np.inf)

    def show_path_local_map(self, trajectory, gt_path, local_map, idx=0, indices=0):
        return write_png(local_map=local_map, center=np.array([local_map.shape[0] / 2, local_map.shape[1] / 2]),
                         file=join(self.output_dir, "local_map_trajectory_{}.png".format(indices + idx)),
                         paths=[self.convert_path_pixel(trajectory=trajectory)],
                         others=self.convert_path_pixel(trajectory=gt_path))

    @torch.no_grad()
    def evaluate(self, input_dict, indices=0):
        ygt = input_dict[DataDict.path]
        y_hat = input_dict[DataDict.prediction]
        if self.train_poses:
            y_hat_poses = y_hat * self.scale_waypoints
        else:
            y_hat_poses = torch.cumsum(y_hat, dim=1) * self.scale_waypoints

        if self.output_dir is not None:
            all_trajectories = input_dict[DataDict.all_trajectories]
            local_map = input_dict[DataDict.local_map]
            for idx in range(len(y_hat_poses)):
                self.show_path_local_map(trajectory=y_hat_poses[idx].detach().cpu().numpy(),
                                         gt_path=ygt[idx].detach().cpu().numpy(),
                                         local_map=local_map[idx].detach().cpu().numpy(), idx=idx, indices=indices)
                if self.train_poses:
                    all_trajectories = [t_hat[idx] * self.scale_waypoints for t_hat in all_trajectories]
                else:
                    all_trajectories = [np.cumsum(t_hat[idx], axis=0) * self.scale_waypoints for t_hat in all_trajectories]
                for t_idx in range(len(all_trajectories)):
                    self.show_path_local_map(trajectory=all_trajectories[t_idx], gt_path=ygt[idx].detach().cpu().numpy(),
                                             local_map=local_map[idx].detach().cpu().numpy(), idx=t_idx, indices=indices)

            path_dis = self.distance(ygt, y_hat_poses).mean()
            last_pose_dis = self.target_dis(ygt[:, -1, :], y_hat_poses[:, -1, :])
            output = {
                LossNames.evaluate_last_dis: last_pose_dis,
                LossNames.evaluate_path_dis: path_dis,
            }

            if self.use_traversability:
                local_map = input_dict[DataDict.local_map]
                traversability_loss = self._local_collision(yhat=y_hat_poses, local_map=local_map)
                traversability_loss_mean = traversability_loss.mean()
                output.update({LossNames.evaluate_traversability: traversability_loss_mean})
            return output


def write_png(local_map=None, rgb_local_map=None, center=None, targets=None, paths=None, paths_color=None, path=None,
              crop_edge=None, others=None, file=None):
    dis = 2
    x_range = [local_map.shape[0], 0]
    y_range = [local_map.shape[1], 0]
    if rgb_local_map is not None:
        local_map_fig = rgb_local_map
    else:
        local_map_fig = np.repeat(local_map[:, :, np.newaxis], 3, axis=2) * 255
    if center is not None:
        assert center.shape[0] == 2 and len(center.shape) == 1, "path should be 2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(center + np.array([x, y]))
        all_points = np.stack(all_points).astype(int)
        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if targets is not None and len(targets) > 0:
        xs, ys = targets[:, 0], targets[:, 1]
        xs = np.clip(xs, dis, local_map_fig.shape[0] - dis)
        ys = np.clip(ys, dis, local_map_fig.shape[1] - dis)
        clipped_targets = np.stack((xs, ys), axis=-1)

        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(clipped_targets + np.array([x, y]))
        if len(clipped_targets.shape) == 2:
            all_points = np.concatenate(all_points, axis=0).astype(int)
        else:
            all_points = np.stack(all_points, axis=0).astype(int)

        local_map_fig[all_points[:, 0], all_points[:, 1], 2] = 0
        local_map_fig[all_points[:, 0], all_points[:, 1], 1] = 255
        local_map_fig[all_points[:, 0], all_points[:, 1], 0] = 0

        if x_range[0] > min(all_points[:, 0]):
            x_range[0] = min(all_points[:, 0])
        if x_range[1] < max(all_points[:, 0]):
            x_range[1] = max(all_points[:, 0])
        if y_range[0] > min(all_points[:, 1]):
            y_range[0] = min(all_points[:, 1])
        if y_range[1] < max(all_points[:, 1]):
            y_range[1] = max(all_points[:, 1])
    if others is not None:
        assert others.shape[1] == 2 and len(others.shape) == 2, "path should be Nx2"
        all_points = []
        for x in range(-dis, dis, 1):
            for y in range(-dis, dis, 1):
                all_points.append(others + np.array([x, y]))
        all_points = np.concatenate(all_points, axis=0).astype(int)

        xs, ys = all_points[:, 0], all_points[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 255
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 0

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if path is not None:
        assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
        all_pts = path
        all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                  all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                  all_pts), axis=0)
        xs, ys = all_pts[:, 0], all_pts[:, 1]
        xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
        ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
        local_map_fig[xs, ys, 0] = 0
        local_map_fig[xs, ys, 1] = 255
        local_map_fig[xs, ys, 2] = 255

        if x_range[0] > min(xs):
            x_range[0] = min(xs)
        if x_range[1] < max(xs):
            x_range[1] = max(xs)
        if y_range[0] > min(ys):
            y_range[0] = min(ys)
        if y_range[1] < max(ys):
            y_range[1] = max(ys)
    if paths is not None:
        for p_idx in range(len(paths)):
            path = paths[p_idx]
            if len(path) == 1 or np.any(path[0] == np.inf):
                continue
            path = np.asarray(path, dtype=int)
            assert path.shape[1] == 2 and len(path.shape) == 2 and path.shape[0] >= 2, "path should be Nx2"
            all_pts = path
            all_pts = np.concatenate((all_pts + np.array([0, -1], dtype=int), all_pts + np.array([1, 0], dtype=int),
                                      all_pts + np.array([-1, 0], dtype=int), all_pts + np.array([0, 1], dtype=int),
                                      all_pts), axis=0)
            xs, ys = all_pts[:, 0], all_pts[:, 1]
            xs = np.clip(xs, 0, local_map_fig.shape[0] - 1)
            ys = np.clip(ys, 0, local_map_fig.shape[1] - 1)
            if paths_color is not None:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 0
                local_map_fig[xs, ys, 2] = paths_color[p_idx]
            else:
                local_map_fig[xs, ys, 0] = 0
                local_map_fig[xs, ys, 1] = 255
                local_map_fig[xs, ys, 2] = 255

            if x_range[0] > min(all_pts[:, 0]):
                x_range[0] = min(all_pts[:, 0])
            if x_range[1] < max(all_pts[:, 0]):
                x_range[1] = max(all_pts[:, 0])
            if y_range[0] > min(all_pts[:, 1]):
                y_range[0] = min(all_pts[:, 1])
            if y_range[1] < max(all_pts[:, 1]):
                y_range[1] = max(all_pts[:, 1])
    if crop_edge:
        local_map_fig = local_map_fig[
                        max(0, x_range[0] - crop_edge):min(x_range[1] + crop_edge, local_map_fig.shape[0]),
                        max(0, y_range[0] - crop_edge):min(y_range[1] + crop_edge, local_map_fig.shape[1])]
    if file is not None:
        cv2.imwrite(file, local_map_fig)
    return local_map_fig
