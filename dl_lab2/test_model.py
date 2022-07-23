import os
import torch
import torch.nn as nn

from main import load_data, evaluate
from eeg_net import EegNet


device = "cuda:7"
train_loader, test_loader = load_data(256)
model = EegNet(nn.ReLU())
model.load_state_dict(torch.load(os.path.join('models', 'eegnet_best_model.pt')))
model.to(device)
acc = evaluate(model, test_loader, device)
print("Test result: {}".format(acc))