import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim))) # [num_ensemble, input_dim, output_dim]
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim))) # [num_ensemble, 1, output_dim]

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5)) # He initialization

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ensemble linear layer. Independent linear transformations for each model in the ensemble on shared or individual inputs.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim) or (num_ensemble, batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (num_ensemble, batch_size, output_dim)."""
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            # x is of shape (batch_size, input_dim) - single input for entire ensemble
            # Feed the input through all models in the ensemble (in practice this is used in the input layer)
            x = torch.einsum('ij,bjk->bik', x, weight) # [num_ensemble, batch_size, output_dim] - res[b,i,k] = SUM_j (x[i,j] * w[b,j,k])
        else:
            # x is of shape (num_ensemble, batch_size, input_dim)
            # Each model in the ensemble gets its own input (which in practice is usually the output of a previous layer)
            x = torch.einsum('bij,bjk->bik', x, weight) # [num_ensemble, batch_size, output_dim] - res[b,i,k] = SUM_j (x[b,i,j] * w[b,j,k])

        x = x + bias # [num_ensemble, batch_size, output_dim]
        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss