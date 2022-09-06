import os
import torch
import torch.nn as nn
import torchvision

class EnsembleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(num_classes=256)
        self.label_fc = nn.Linear(3, 256)
        self.fc = nn.Linear(256 + 256, 2)

    def forward(self, x, label):
        x = self.net(x)
        label = self.label_fc(label)
        x = torch.cat([x, label], dim=-1)
        x = self.fc(x)
        return x

