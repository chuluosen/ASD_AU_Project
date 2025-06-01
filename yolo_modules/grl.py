# yolo_modules/grl.py
import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int = 256, lambd: float = 0.1):
        super().__init__()
        self.lambd = lambd
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
    
    def forward(self, feat):
        feat = GradientReverse.apply(feat, self.lambd)
        return self.net(feat)