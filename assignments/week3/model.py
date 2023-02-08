import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    MLP class with initialization and forward method. inherits nn.Module
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        Returns: None
        """
        super(MLP, self).__init__()

        # initialize layers of MLP
        self.layers = torch.nn.ModuleList()
        self.batchLayer = torch.nn.ModuleList()
        # increasing hidden units
        for i in range(hidden_count // 2 + 1):
            self.layers += [torch.nn.Linear(input_size, hidden_size)]
            # print("now in ",i," ",input_size," ",hidden_size)
            input_size = hidden_size
            hidden_size *= 2
            # initialize weight
            initializer(self.layers[-1].weight)
            # add batch norm
            self.batchLayer += [torch.nn.BatchNorm1d(input_size)]
        # then decreasing hidden units
        for i in range(hidden_count // 2 + 1, hidden_count):
            hidden_size = hidden_size // 4
            self.layers += [torch.nn.Linear(input_size, hidden_size)]
            # print("now in ",i," ",input_size," ",hidden_size)
            input_size = hidden_size
            hidden_size = hidden_size // 2
            # initialize weight
            initializer(self.layers[-1].weight)
            # add batch norm
            self.batchLayer += [torch.nn.BatchNorm1d(input_size)]

        # output layer and initilize weight
        self.out = torch.nn.Linear(input_size, num_classes)
        initializer(self.out.weight)

        self.activation = activation
        self.initializer = initializer
        self.hidden_count = hidden_count
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # go through the layers
        for i in range(self.hidden_count):
            x = self.layers[i](x)
            x = self.batchLayer[i](x)
            x = self.activation(x)

        # output layer
        x = self.dropout(x)
        x = self.out(x)

        return x
