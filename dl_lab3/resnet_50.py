import torch
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights


class ResNet50(nn.Module):

    def __init__(self, pretrained: bool, requires_grad=True, num_classes=5):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        self.set_requires_grad(requires_grad)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction_y = self.resnet50(x)
        return prediction_y

    def set_requires_grad(self, requires_grad):
        for param in self.resnet50.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def get_name():
        return "ResNet50"
