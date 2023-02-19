from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, factor=0.9, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.factor = factor
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # print(type(self.last_epoch))

        if self.last_epoch == 0:
            return [i for i in self.base_lrs]
        return [i * np.exp(-self.factor * self.last_epoch) for i in self.base_lrs]

        # return [i*j for (i, j) in zip(self.base_lrs , decay)]
        # return [group["lr"] * self.factor for group in self.optimizer.param_groups]
