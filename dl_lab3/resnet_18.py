import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights


class ResNet18(nn.Module):

    def __init__(self, pretrained: bool, requires_grad=True, num_classes=5):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        self.resnet18 = torchvision.models.resnet18(weights=weights)
        self.set_requires_grad(requires_grad)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction_y = self.resnet18(x)
        return prediction_y

    def set_requires_grad(self, requires_grad):
        for param in self.resnet18.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def get_name():
        return "ResNet18"
