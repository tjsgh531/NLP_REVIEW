import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, sr_model, output_dim, vocab_size, embed_dim, **kwargs) :
        super().__init__()
        self.sr_model = sr_model(vocab_size = vocab_size, embed_dim= embed_dim, **kwargs)

        self.input_dim = self.sr_model.output_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc(self.sr_model(x))
    
