import torch.nn as nn
import torch.nn.init as init
import torch

class Model(nn.Module):
    def __init__(self, input_dim = 2, hidden_size= 32, output_dim=2) :
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_dim)

        # dropout
        self.dropout = nn.Dropout(p = 0.2)

        # normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # initialization(uniform)
        init.xavier_uniform_(self.linear_1.weight.data)
        init.xavier_uniform_(self.linear_2.weight.data)

        # initialization(normalization)
        self.apply(self.__init_weight)

    def __init_weight(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)

    def forward(self, x):
        h = torch.relu(self.linear_1(x))
        logit = self.linear_2(h)
        probabilities = torch.softmax(logit, dim=1)

        return probabilities