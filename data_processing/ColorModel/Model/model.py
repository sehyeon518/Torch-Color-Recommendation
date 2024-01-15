import torch
import torch.nn as nn

dropout_rate = 0.5
alpha = 0.001

class LassoModel(nn.Module):
    def __init__(self, input_size):
        super(LassoModel, self).__init__()
        self.normalize1 = nn.LayerNorm((input_size,))
        self.hidden1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.normalize2 = nn.LayerNorm((64,))
        self.hidden2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(32, 12)

    def forward(self, x):
        x = self.normalize1(x)
        x = torch.nn.functional.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.normalize2(x)
        x = torch.nn.functional.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.linear(x)
        return x
    
def lasso_loss(model, h, y):
    mse_loss = nn.functional.mse_loss(h, y)
    
    l1_regularization = alpha * torch.sum(torch.abs(model.linear.weight))
    
    total_loss = mse_loss + l1_regularization
    
    return total_loss