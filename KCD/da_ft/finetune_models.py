import torch
from torch import nn
import torch.nn.functional as F
from fmcib.models import fmcib_model

class LinearClassifier(nn.Module):
    def __init__(self,  output_dim):
        super(LinearClassifier, self).__init__()
        self.feature_extractor = fmcib_model()
        self.fc = nn.Linear(4096, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out

