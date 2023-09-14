import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.output_dim = embed_dim

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx= 0)

    def forward(self, x):
        x_embed = self.embeddings(x)

        stnc_repr = torch.mean(x_embed, dim=1)

        return stnc_repr