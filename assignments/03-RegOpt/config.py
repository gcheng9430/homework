from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor

# import torchvision.transforms as transforms


class CONFIG:
    batch_size = 64
    num_epochs = 10
    initial_learning_rate = 0.0025
    initial_weight_decay = 0.001

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "num_epochs": num_epochs,
        "factor": 0.001,
        "initial_learning_rate": initial_learning_rate,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
