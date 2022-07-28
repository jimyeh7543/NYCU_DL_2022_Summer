import os
import torch
import torch.nn as nn

from main import load_data, evaluate
from resnet_50 import ResNet50


device = "cuda:7"
train_loader, test_loader = load_data(256)
model = ResNet50(pretrained=None, requires_grad=False)
model.load_state_dict(torch.load(os.path.join('models', 'resnet50_8228_best_model.pt')))
model.to(device)
acc, labels = evaluate(model, test_loader, device)
print("Test result: {}".format(acc))