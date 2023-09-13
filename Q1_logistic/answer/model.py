import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_feature = 9):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(n_feature))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        logit = torch.sum(x * self.weight + self.bias, dim=1)

        return torch.sigmoid(logit)