import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d
from stnet import stnet50


from coordconv import CoordConv2d


class STResnextBirds(nn.Module):
    def __init__(self):
        super(STResnextBirds, self).__init__()
        self.stnet = stnet50(input_channels=3, num_classes=200, T=1, N=1)

    def forward(self, x):
        # Perform the usual forward pass
        x = self.stnet(x)
        return F.log_softmax(x, dim=1)


class ResnextBirds(nn.Module):
    def __init__(self):
        super(ResnextBirds, self).__init__()
        self.resnext50 = resnext50_32x4d(pretrained=False)

    def forward(self, x):
        # Perform the usual forward pass
        x = self.resnext50(x)
        return F.log_softmax(x, dim=1)