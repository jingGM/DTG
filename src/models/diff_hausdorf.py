import torch
from torch import nn
from src.utils.configs import Hausdorff


class HausdorffLoss(nn.Module):
    def __init__(self, mode=Hausdorff.average):
        super(HausdorffLoss, self).__init__()
        self.mode = mode

    def forward(self, set1, set2):
        if len(set1.shape) == 2:
            set1 = set1.unsqueeze(0)
        if len(set2.shape) == 2:
            set2 = set2.unsqueeze(0)
        assert len(set1.shape) == 3 and len(set2.shape) == 3, "the shape should be B x N x C"
        assert set1.shape[0] == set2.shape[0] and set1.shape[2] == set2.shape[2], \
            "the B and C should be the same in two sets"
        # B,M,C = set1.shape
        # B,N,C = set2.shape
        set1 = set1.unsqueeze(2)
        set2 = set2.unsqueeze(1)
        distances = torch.norm(set1 - set2, dim=-1, p=2)  # B x M x N
        if self.mode == Hausdorff.average:
            term_1 = torch.mean(torch.min(distances, 2)[0], dim=1)  # B x M -> B
            term_2 = torch.mean(torch.min(distances, 1)[0], dim=1)  # B x N -> B
        elif self.mode == Hausdorff.max:
            term_1 = torch.max(torch.min(distances, 2)[0], dim=1)[0]  # B x M -> B
            term_2 = torch.max(torch.min(distances, 1)[0], dim=1)[0]  # B x N -> B
        else:
            raise Exception("Hausdorff mode is not defined.")
        res = term_1 + term_2
        if res.shape[0] == 1:
            res = res.squeeze(0)
        return res
