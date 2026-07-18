from torch import nn
from torchvision import models
def create_model(pretrained=True):
    m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,2); return m
def freeze_backbone(m):
    for p in m.features.parameters(): p.requires_grad=False
def unfreeze_final_blocks(m,n=2):
    for p in m.features[-n:].parameters(): p.requires_grad=True
