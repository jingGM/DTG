import copy
import os
import pickle
from os.path import join
import random
from random import shuffle
import open3d as o3d
import numpy as np
import warnings
from numba import jit

from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.configs import DataDict
from src.utils.functions import inverse_transform


@jit(nopython=True)
def process_single_lidar(line_pts, distance, lidar_horizons, lidar_angle_range, angle_to_idx, lidar_threshold):
    line_values = np.zeros(lidar_horizons)
    for i in range(len(line_pts)):
        pt = line_pts[i]
        dis = distance[i]
        if dis > lidar_threshold:
            dis = 0
        idx = int((np.arctan2(pt[1], pt[0]) + lidar_angle_range / 2.0) * angle_to_idx)
        if 0 < idx < lidar_horizons:
            if line_values[idx] <= 0.5:
                line_values[idx] = dis
            else:
                if line_values[idx] > dis:
                    line_values[idx] = dis
    return line_values


class TrainData(Dataset):
    def __init__(self, cfg, train: bool):
        with open(join(cfg.root, "data.pkl"), "rb") as input_file:
            data = pickle.load(input_file)
        self.original_files = data[DataDict.file_names]
        self.data_root = join(cfg.root, "data_folder")
        self.train = train
        self.data_percentage = cfg.training_data_percentage

        self.lidar_threshold = cfg.lidar_cfg.threshold
        self.lidar_channels = cfg.lidar_cfg.channels
        self.lidar_horizons = cfg.lidar_cfg.horizons
        self.lidar_angle_range = cfg.lidar_cfg.angle_range * np.pi / 180.0
        self.angle_to_idx = float(self.lidar_horizons) / self.lidar_angle_range

        self.vel_num = cfg.vel_num
        self.imu_num = cfg.imu_num

        self.all_files = []
        self._process_files()

    def _process_files(self):
        all_files = []
        for fn, tnum in self.original_files:
            for i in range(tnum):
                all_files.append((fn, i))
        random.seed(0)
        shuffle(all_files)
        select_split = int(len(all_files) * self.data_percentage)
        if self.train:
            self.all_files = copy.deepcopy(all_files[:select_split])
        else:
            self.all_files = copy.deepcopy(all_files[select_split:])

    def load_data(self, index):
        fn, idx = self.all_files[index]
        with open(join(self.data_root, fn), "rb") as input_file:
            current_data = pickle.load(input_file)
        velocity = current_data[DataDict.vel]
        imu = current_data[DataDict.imu]
        target, heuristic = current_data[DataDict.targets][idx]
        lidar = current_data[DataDict.lidar]
        camera = current_data[DataDict.camera]
        local_map = current_data[DataDict.local_map]
        trajectory = current_data[DataDict.trajectories][idx]
        pose = current_data[DataDict.pose]

        relative_target = inverse_transform(pts=target[:3], transformation=current_data[DataDict.pose])[0, :2]
        return lidar, camera, imu, velocity, trajectory, relative_target, heuristic, local_map, pose, target

    def __len__(self):
        return len(self.all_files)

    def _process_lidar(self, points):
        # points = np.concatenate((points[:, :3], np.zeros_like(points[:, :1])), axis=-1)
        lidar_array = np.zeros((self.lidar_channels, self.lidar_horizons))
        for i in range(self.lidar_channels):
            line_pts = points[np.where(points[:, 4] == i, True, False)][:, :3]
            distance = np.linalg.norm(line_pts, axis=1)
            lidar_array[i] = process_single_lidar(distance=distance, lidar_threshold=self.lidar_threshold,
                                                  lidar_horizons=self.lidar_horizons, angle_to_idx=self.angle_to_idx,
                                                  line_pts=line_pts, lidar_angle_range=self.lidar_angle_range)
        return lidar_array

    def __getitem__(self, index):
        lidar, camera, imu, velocity, trajectory, target, heuristic, local_map, pose, raw_target = self.load_data(index=index)
        if self.lidar_threshold is not None:
            lidar_data = np.clip(np.asarray([self._process_lidar(pts) for pts in lidar]) / self.lidar_threshold, 0.0, 1.0)
        else:
            lidar_data = np.asarray(lidar)
        output_dict = {
            DataDict.lidar: lidar_data,
            DataDict.target: target,
            DataDict.raw_target: raw_target,
            DataDict.heuristic: heuristic,
            DataDict.path: trajectory,
            DataDict.local_map: local_map,
            DataDict.pose: pose,
            DataDict.camera: camera,
        }
        if self.vel_num > 0:
            output_dict.update({DataDict.vel: velocity[:self.vel_num]})
        if self.imu_num > 0:
            output_dict.update({DataDict.imu: imu[:self.imu_num]})
        return output_dict
