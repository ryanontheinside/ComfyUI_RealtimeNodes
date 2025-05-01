import torch
import torch.nn as nn
import torch.nn.functional as F

class RealTimeFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight CNN for real-time flow estimation
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)
        
        # Initialize weights for better flow estimation
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        # x should be in BCHW format
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x