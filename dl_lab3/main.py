import sys
import logging
import argparse

from tqdm import tqdm
from typing import Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as op
from torch import device
from torch.utils.data import DataLoader

from utils import save_model
from resnet_18 import ResNet18
from resnet_50 import ResNet50
from visualization import save_result, save_confusion_matrix
from dataloader import RetinopathyLoader

logging.basicConfig(filename='dl_lab3.log', level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger = logging.getLogger("dl_lab3")
logger.addHandler(handler)


def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = RetinopathyLoader("retinopathy_data", "train")
    test_dataset = RetinopathyLoader("retinopathy_data", "test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    return train_loader, test_loader


def train(model: nn.Module, optimizer: op, loss_func: nn.modules.loss, train_loader: DataLoader,
          device: device) -> float:
    acc = 0
    model.train()
    for train_images, train_labels in train_loader:
        train_images = train_images.to(device)
        train_labels = train_labels.to(device, dtype=torch.long)
        prediction_labels = model.forward(train_images)
        optimizer.zero_grad()
        loss = loss_func(prediction_labels, train_labels)
        loss.backward()
        optimizer.step()
        acc += prediction_labels.max(dim=1)[1].eq(train_labels).sum().item()
    acc = 100.0 * acc / len(train_loader.dataset)
    return round(acc, 2)


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: device):
    model.eval()
    acc = 0
    labels = {"ground_truth": [], "prediction": []}
    for test_data, test_labels in test_loader:
        test_data = test_data.to(device)
        test_labels = test_labels.to(device, dtype=torch.long)

        prediction_labels = model.forward(test_data)
        predict_class = prediction_labels.max(dim=1)[1]
        acc += predict_class.eq(test_labels).sum().item()
        labels['ground_truth'].extend(test_labels.tolist())
        labels['prediction'].extend(predict_class.tolist())
    acc = 100.0 * acc / len(test_loader.dataset)
    return round(acc, 2), labels


def select_best_model(model_class: nn.Module, n_epochs: int, n_epochs_classifier: int, batch_size: int,
                      learning_rate: float, learning_rate_classifier: float, momentum: float, weight_decay: float,
                      device: device):
    train_loader, test_loader = load_data(batch_size)
    pretrain = {"with pretraining": True, "w/o pretraining": False}
    acc_history = {"Train": {}, "Test": {}}
    max_accuracy = 0
    best_classifier = None
    best_model_state = None
    best_labels = None
    for key, value in pretrain.items():
        print("Start to run {} {}".format(model_class.get_name(), key))

        acc_history["Train"][key] = [0] * (n_epochs + n_epochs_classifier)
        acc_history["Test"][key] = [0] * (n_epochs + n_epochs_classifier)
        model = model_class(pretrained=value, requires_grad=False)
        optimizer = op.SGD(model.parameters(), lr=learning_rate_classifier, momentum=momentum,
                           weight_decay=weight_decay)
        loss_func = nn.CrossEntropyLoss()
        model.to(device)
        if n_epochs_classifier > 0:
            print("Training for classifier")
            for epoch in tqdm(range(0, n_epochs_classifier)):
                acc_history["Train"][key][epoch] = train(model, optimizer, loss_func, train_loader, device)
                acc_history["Test"][key][epoch], labels = evaluate(model, test_loader, device)
                if acc_history["Test"][key][epoch] > max_accuracy:
                    max_accuracy = acc_history["Test"][key][epoch]
                    best_model_state = deepcopy(model.state_dict())
                    best_classifier = model

        model = best_classifier
        model.set_requires_grad(True)
        optimizer = op.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        for epoch in tqdm(range(0, n_epochs)):
            acc_index = n_epochs_classifier + epoch
            acc_history["Train"][key][acc_index] = train(model, optimizer, loss_func, train_loader, device)
            acc_history["Test"][key][acc_index], labels = evaluate(model, test_loader, device)
            if acc_history["Test"][key][acc_index] > max_accuracy:
                max_accuracy = acc_history["Test"][key][acc_index]
                best_labels = labels
                best_model_state = deepcopy(model.state_dict())
    return acc_history, max_accuracy, best_model_state, best_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_epochs", default=10, type=int)
    parser.add_argument("-nc", "--n_epochs_classifier", default=10, type=int)
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("-l", "--learning_rate", default=5e-4, type=float)
    parser.add_argument("-lc", "--learning_rate_classifier", default=1e-3, type=float)
    parser.add_argument("-m", "--momentum", default=0.9, type=float)
    parser.add_argument("-w", "--weight_decay", default=5e-4, type=float)
    parser.add_argument("-d", "--device", default="cuda:7", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    model_class = ResNet18

    acc_history, max_acc, model_state, labels = select_best_model(model_class,
                                                                  args.n_epochs,
                                                                  args.n_epochs_classifier,
                                                                  args.batch_size,
                                                                  args.learning_rate,
                                                                  args.learning_rate_classifier,
                                                                  args.momentum,
                                                                  args.weight_decay,
                                                                  args.device)
    logger.info("The best test result of {}:".format(model_class.get_name()))
    for key, value in acc_history["Test"].items():
        logger.info("    {}: {}".format(key, max(value)))
    max_acc_str = str(max_acc).replace('.', '')
    save_model("{}_{}".format(model_class.get_name().lower(), max_acc_str), model_state)
    save_result(model_class.get_name(), max_acc_str, acc_history)
    save_confusion_matrix(model_class.get_name(), max_acc_str, labels["ground_truth"], labels['prediction'],
                          ["0", "1", "2", "3", "4"])


if __name__ == '__main__':
    main()
