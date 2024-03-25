import torch

from src.train import Trainer
from src.utils.arguments import get_configuration


if __name__ == "__main__":
    cfgs = get_configuration()
    trainer = Trainer(cfgs=cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.run()
    torch.autograd.set_detect_anomaly(False)
