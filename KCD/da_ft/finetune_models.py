import torch
from torch import nn
import torch.nn.functional as F
from fmcib.models import fmcib_model

class LinearClassifier(nn.Module):
    def __init__(self,  output_dim):
        super(LinearClassifier, self).__init__()
        self.feature_extractor = fmcib_model()
        self.fc = nn.Linear(4096, output_dim)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        with torch.no_grad():
            x = self.dropout(self.feature_extractor(x))
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out

class NonLinearClassifier(nn.Module):
    def __init__(self,  output_dim,h=64):
        super(NonLinearClassifier, self).__init__()
        self.feature_extractor = fmcib_model()
        self.fc1 = nn.Linear(4096, h*4)
        self.fc2 = nn.Linear(h*4, h)
        self.out = nn.Linear(h, output_dim)
        self.dropout = nn.Dropout(0.8)
        self.actv = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            x = self.dropout(self.feature_extractor(x))
        x = self.actv(self.fc1(x))
        x = self.actv(self.fc2(x))
        out = F.softmax(self.out(x), dim=1)
        return out

class EndToEndClassifier(nn.Module):
    def __init__(self,  output_dim):
        super(EndToEndClassifier, self).__init__()
        self.feature_extractor = fmcib_model()
        self.fc = nn.Linear(4096, output_dim)
        self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        x = self.dropout(self.feature_extractor(x))
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out

