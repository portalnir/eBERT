import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for i in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for i in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for i in range(num_layers)])
        self.f = f

    def forward(self, input):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](input))
            nonlinear = self.f(self.nonlinear[layer](input))
            linear = self.linear[layer](input)
            input = gate * nonlinear + (1 - gate) * linear

        return input
