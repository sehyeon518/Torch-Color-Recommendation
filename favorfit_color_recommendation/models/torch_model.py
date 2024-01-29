import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x):
        residue = x
        x = self.layer1(x)
        x = self.layer2(x)
        return x + residue
    
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.seq_modules = nn.Sequential(
            nn.Linear(119, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )

        self.out = nn.Linear(1024, 540)

    def forward(self, x):
        x = self.seq_modules(x)
        x = self.out(x)
        return x