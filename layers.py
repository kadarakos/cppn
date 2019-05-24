import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.normal(self.weight, 0, std)
        #nn.init.uniform(self.bias, -std, std)

    def forward(self, input):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        bias = self.bias
        if bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)

class CustomActivationFunction(nn.Module):

    def __init__(self, mean=0, std=1, min=-0.9, max=0.9):
        super(CustomActivationFunction, self).__init__()
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.gauss = lambda x: torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))
        self.cos = torch.cos
        self.func = np.random.choice([self.gauss, self.cos])
    
    def forward(self, x):
        x = self.func(x)
        return torch.clamp(x, min=self.min, max=self.max)


