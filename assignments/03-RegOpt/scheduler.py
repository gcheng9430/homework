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
        self.total_iters = self.num_epochs * 782
        self.min_lrs = [0] * len(optimizer.param_groups)
        self.num_bad_epochs = 0
        self.best = None
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def _reduce_lr(self) -> None:
        """
        Arguments:None
        Returns: None
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * 0.01, self.min_lrs[i])
            if old_lr - new_lr > 1e-8:
                param_group["lr"] = new_lr

    # def is_better(self, curr, best):
    #     self.threshold = 1e-4
    #     rel_epsilon = 1. - self.threshold
    #     return curr < best * rel_epsilon

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

        # # this is time based decay
        # decay = self.initial_learning_rate / self.num_epochs
        # return [i * 1 / (1 + decay * self.last_epoch) for i in self.base_lrs]

        # #reduce lr on plateau
        # current = float(metrics)   #cannot pass in metrics thus cannot use reducelrplateau
        # if self.is_better(current, self.best):
        #     self.best = current
        #     self.num_bad_epochs = 0
        # else:
        #     self.num_bad_epochs += 1

        # if self.num_bad_epochs > 10:
        #     self._reduce_lr()
        #     self.num_bad_epochs = 0
        #     return [group["lr"] for group in self.optimizer.param_groups]

        if self.last_epoch >= 5000:
            self._reduce_lr()
            return [
                0.00001
                + (i - 0.00001)
                * 1
                * (1 + np.cos(np.pi * self.last_epoch / self.total_iters))
                / 2
                for i in self.base_lrs
            ]

        # if self.last_epoch >= 3000:
        #     return [i * np.exp(-self.factor * self.last_epoch ** 0.8) for i in self.base_lrs]

        # This is expo decay - current best
        # return [i * np.exp(-self.factor * self.last_epoch ** 1) for i in self.base_lrs]

        # This is linear decay
        # return [i  * (0.999 + (0.02 - 0.999) * min(self.total_iters, self.last_epoch)
        # / self.total_iters) for i in self.base_lrs]

        # return [i*j for (i, j) in zip(self.base_lrs , decay)]
        # return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        # this is cosineanneling
        return [
            0 + (i - 0) * (1 + np.cos(np.pi * self.last_epoch / self.total_iters)) / 2
            for i in self.base_lrs
        ]
