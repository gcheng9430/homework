from typing import Callable
import torch


class MLP(torch.nn.Module):
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
        """
        super(MLP,self).__init__()

        #initialize layers of MLP
        self.layers = torch.nn.ModuleList()
        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(input_size,hidden_size)]
            input_size = hidden_size
            #initialize weight
            initializer(self.layers[-1].weight)
        #output layer and initilize weight 
        self.out = torch.nn.Linear(hidden_size,num_classes)
        initializer(self.out.weight)
        
        self.activation = activation
        self.initializer = initializer


    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.out(x)

        return x

