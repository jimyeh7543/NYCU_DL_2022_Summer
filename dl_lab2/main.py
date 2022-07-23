import sys
import logging
import argparse
from typing import Tuple
from copy import deepcopy

import torch
import torch.optim as op
import torch.nn as nn
from torch import device
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from dataloader import read_bci_data
from eeg_net import EegNet
from deepconv_net import DeepConvNet
from visualization import save_result
from utils import save_model

logging.basicConfig(filename='dl_lab2.log', level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root = logging.getLogger()
root.addHandler(handler)


def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader


def train(model: nn.Module, optimizer: op, loss_func: nn.modules.loss, train_loader: DataLoader,
          device: device) -> float:
    acc = 0
    model.train()
    for train_data, train_labels in train_loader:
        train_data = train_data.to(device)
        train_labels = train_labels.to(device, dtype=torch.long)

        prediction_labels = model.forward(train_data)
        optimizer.zero_grad()
        loss = loss_func(prediction_labels, train_labels)
        loss.backward()
        optimizer.step()
        acc += prediction_labels.max(dim=1)[1].eq(train_labels).sum().item()
    acc = 100.0 * acc / len(train_loader.dataset)
    return round(acc, 2)


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: device) -> float:
    model.eval()
    acc = 0
    for test_data, test_labels in test_loader:
        test_data = test_data.to(device)
        test_labels = test_labels.to(device, dtype=torch.long)
        prediction_labels = model.forward(test_data)
        acc += prediction_labels.max(dim=1)[1].eq(test_labels).sum().item()
    acc = 100.0 * acc / len(test_loader.dataset)
    return round(acc, 2)


def select_best_model(model_class: nn.Module, n_epochs: int, batch_size: int, learning_rate: float, weight_decay: float,
                      device: device):
    train_loader, test_loader = load_data(batch_size)
    activation_funcs = {"ELU": nn.ELU(alpha=1.0), "ReLU": nn.ReLU(), "LeakyReLU": nn.LeakyReLU()}
    # activation_funcs = {"ReLU": nn.ReLU()}
    acc_history = {"Train": {}, "Test": {}}
    max_accuracy = {}
    overall_max_accuracy = 0
    best_model_state = None
    for key, activation_func in activation_funcs.items():
        print("Start to run {} with {}".format(model_class.get_name(), key))
        acc_history["Train"][key] = [0] * n_epochs
        acc_history["Test"][key] = [0] * n_epochs
        max_accuracy[key] = 0
        model = model_class(activation_func)
        optimizer = op.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_func = nn.CrossEntropyLoss()
        model.to(device)
        for epoch in tqdm(range(0, n_epochs)):
            acc_history["Train"][key][epoch] = train(model, optimizer, loss_func, train_loader, device)
            acc_history["Test"][key][epoch] = evaluate(model, test_loader, device)
            if acc_history["Test"][key][epoch] > max_accuracy[key]:
                max_accuracy[key] = acc_history["Test"][key][epoch]
                if max_accuracy[key] > overall_max_accuracy:
                    overall_max_accuracy = max_accuracy[key]
                    best_model_state = deepcopy(model.state_dict())
    return acc_history, max_accuracy, best_model_state, overall_max_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_epochs", default=300, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-l", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-w", "--weight_decay", default=1e-2, type=float)
    parser.add_argument("-d", "--device", default="cuda:7", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info(args)
    model_class = EegNet
    acc_history, max_accuracy, model_state, model_accuracy = select_best_model(model_class,
                                                                               args.n_epochs,
                                                                               args.batch_size,
                                                                               args.learning_rate,
                                                                               args.weight_decay,
                                                                               args.device)

    logging.info("The best test result of {}:".format(model_class.get_name()))
    for key, value in max_accuracy.items():
        logging.info("    {}: {}".format(key, value))
    model_name = "{}_{}".format(model_class.get_name().lower(), str(model_accuracy).replace('.', ''))
    save_model(model_name, model_state)
    save_result(model_class.get_name(), str(model_accuracy).replace('.', ''), acc_history)
