from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class CustomLRScheduler(_LRScheduler):
    """
    Learning rate Scheduler class that adjust learning rate
    """

    def __init__(
        self, optimizer, num_epochs, initial_learning_rate, factor=0.9, last_epoch=-1
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.factor = factor
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Arguments:
        None
        Returns:
        List[float] the learning rate for this epoch
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # print(type(self.last_epoch))

        # print(self.last_epoch)
        if self.last_epoch == 0:
            return [i for i in self.base_lrs]
        # if self.last_epoch >= 2346:
        #     self.factor *= 2**(1/782)

        # # this is time based decay
        # decay = self.initial_learning_rate / self.num_epochs
        # return [i * 1 / (1 + decay * self.last_epoch) for i in self.base_lrs]

        if self.last_epoch >= 3000:
            return [
                i * np.exp(-self.factor * (self.last_epoch) ** 1.01)
                for i in self.base_lrs
            ]

        # This is expo decay - current best
        return [i * np.exp(-self.factor * self.last_epoch) for i in self.base_lrs]

        # return [i*j for (i, j) in zip(self.base_lrs , decay)]
        # return [group["lr"] * self.factor for group in self.optimizer.param_groups]
