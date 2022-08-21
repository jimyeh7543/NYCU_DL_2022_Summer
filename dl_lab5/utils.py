import os
import torch


def save_model(model_path: str, model_name: str, model_state: dict) -> None:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model_state, os.path.join(model_path, '{}_best_model.pt'.format(model_name)))
