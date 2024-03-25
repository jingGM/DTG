import torch
from torch import nn

from src.models.perception import Perception, LidarImageModel
from src.models.vae import CVAE
from src.models.diffusion import Diffusion
from src.utils.configs import DataDict, GeneratorType


class HNav(nn.Module):
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device

        self.generator_type = config.generator_type
        if self.generator_type == GeneratorType.cvae:
            self.perception = Perception(self.config.perception)
            self.generator = CVAE(self.config.cvae)
        elif self.generator_type == GeneratorType.diffusion:
            self.perception = Perception(self.config.perception)
            self.generator = Diffusion(self.config.diffusion)
        else:
            raise "the generator type is not defined"

    def forward(self, input_dict, sample=False):
        if sample:
            return self.sample(input_dict=input_dict)
        else:
            output = {DataDict.path: input_dict[DataDict.path],
                      DataDict.heuristic: input_dict[DataDict.heuristic],
                      DataDict.local_map: input_dict[DataDict.local_map]}

            observation = self.perception(lidar=input_dict[DataDict.lidar], vel=input_dict[DataDict.vel],
                                          target=input_dict[DataDict.target])
            generator_output = self.generator(observation=observation, gt_path=input_dict[DataDict.path])
            output.update(generator_output)
            return output

    def sample(self, input_dict):
        output = {}
        if DataDict.path in input_dict.keys():
            output.update({DataDict.path: input_dict[DataDict.path]})
        if DataDict.heuristic in input_dict.keys():
            output.update({DataDict.heuristic: input_dict[DataDict.heuristic]})
        if DataDict.local_map in input_dict.keys():
            output.update({DataDict.local_map: input_dict[DataDict.local_map]})
        observation = self.perception(lidar=input_dict[DataDict.lidar], vel=input_dict[DataDict.vel],
                                      target=input_dict[DataDict.target])
        generator_output = self.generator.sample(observation=observation)
        output.update(generator_output)
        return output


def get_model(config, device):
    return HNav(config=config, device=device)
