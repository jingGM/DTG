import time
import warnings
import torch
from torch import nn
from src.utils.configs import DataDict


class LidarImageModel(nn.Module):
    def __init__(self, input_channel=3, lidar_out_dim=512, norm_layer=True):
        super(LidarImageModel, self).__init__()
        if norm_layer:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([8, 12, 910]),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([16, 8, 453]),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2), nn.LayerNorm([32, 6, 226]),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(), nn.LayerNorm([32, 2, 112]),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2), nn.LayerNorm([2048]),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.LayerNorm([1024]),
                nn.Linear(1024, lidar_out_dim), nn.ELU(), nn.LayerNorm([lidar_out_dim]),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=(1, 2)), nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(2, 2)), nn.ELU(),
                nn.Flatten(), nn.Linear(7168, 2048), nn.LeakyReLU(0.2),
                nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, lidar_out_dim), nn.ELU()
            )

    def forward(self, image):
        output = self.conv(image)
        return output


class Perception(nn.Module):
    def __init__(self, cfg):
        super(Perception, self).__init__()
        self.cfg = cfg

        self.lidar_model = LidarImageModel(input_channel=self.cfg.lidar_num, lidar_out_dim=self.cfg.lidar_out,
                                           norm_layer=self.cfg.lidar_norm_layer)

        self.vel_model = nn.Sequential(
            nn.Linear(self.cfg.vel_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, self.cfg.vel_out), nn.LeakyReLU(0.2)
        )

        combo_input_dim = self.cfg.vel_out + self.cfg.lidar_out + 2
        self.combo_layers = nn.Sequential(
            nn.Linear(combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, 2 * combo_input_dim), nn.ELU(),
            nn.Linear(2 * combo_input_dim, combo_input_dim), nn.LeakyReLU(0.2)
        )

    def forward(self, lidar, vel, target):
        lidar_fts = self.lidar_model(lidar)  # B x 512

        VB, VN, VD = vel.size()
        vel_fts = self.vel_model(vel.view(VB, -1))  # B x 256

        observation = torch.concat((lidar_fts, vel_fts, target), dim=1)  # B x 770

        perception = self.combo_layers(observation)
        return perception
