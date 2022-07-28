import os

import torch


def save_model(model_name: str, model_state: dict) -> None:
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model_state, os.path.join('models', '{}_best_model.pt'.format(model_name)))
