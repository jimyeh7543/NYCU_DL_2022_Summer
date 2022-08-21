import json
import os

import numpy as np
import torch
from PIL import Image

from torch.utils import data
from torchvision import transforms


class IclevrDataset(data.Dataset):
    def __init__(self, root, is_train, data_filename="train.json"):
        self.root = root
        self.is_train = is_train

        self.condition_map = json.load(open(os.path.join(root, "objects.json")))
        self.data_json = json.load(open(os.path.join(root, data_filename)))
        self.one_hot_vectors = []

        if self.is_train:
            self.image_names = list(self.data_json.keys())
            for i in range(len(self.image_names)):
                condition = self.data_json[self.image_names[i]]
                one_hot_vector = np.zeros(24, dtype=np.int)
                for j in condition:
                    one_hot_vector[self.condition_map[j]] = 1
                self.one_hot_vectors.append(one_hot_vector)

        else:
            for condition in self.data_json:
                one_hot_vector = np.zeros(24, dtype=np.int)
                for j in condition:
                    one_hot_vector[self.condition_map[j]] = 1
                self.one_hot_vectors.append(one_hot_vector)

        self.preprocessing = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print("> Found %d datas..." % (len(self.data_json)))

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, index):
        if self.is_train:
            path = os.path.join(self.root, "iclevr", self.image_names[index])
            return self.preprocessing(Image.open(path).convert('RGB')), torch.Tensor(self.one_hot_vectors[index])
        return "", torch.Tensor(self.one_hot_vectors[index])
