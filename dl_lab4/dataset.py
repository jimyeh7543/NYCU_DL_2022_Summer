import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
    transforms.ToTensor(),
])


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root = '{}/{}'.format(args.data_root, mode)
        self.seq_len = args.n_past + args.n_future
        self.mode = mode
        self.transform = transform

        self.dirs = []
        for i in os.listdir(self.root):
            if i.startswith('.'):
                continue
            for j in os.listdir(os.path.join(self.root, i)):
                if j.startswith('.'):
                    continue
                self.dirs.append(os.path.join(self.root, i, j))

        self.seed_is_set = False
        self.current_dir = self.dirs[0]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)

    def get_seq(self, index):
        self.current_dir = self.dirs[index]

        image_seq = []
        for i in range(self.seq_len):
            filename = "{}/{}.png".format(self.current_dir, i)
            image_seq.append(self.transform(Image.open(filename)))
        image_seq = torch.stack(image_seq)

        return image_seq

    def get_csv(self):
        with open(os.path.join(self.current_dir, "actions.csv"), newline='') as csv_file:
            rows = csv.reader(csv_file)
            actions = []
            for i, row in enumerate(rows):
                if i == self.seq_len:
                    break
                action = [float(value) for value in row]
                actions.append(torch.tensor(action))
            actions = torch.stack(actions)

        with open(os.path.join(self.current_dir, "endeffector_positions.csv"), newline='') as csv_file:
            rows = csv.reader(csv_file)
            positions = []
            for i, row in enumerate(rows):
                if i == self.seq_len:
                    break
                position = [float(value) for value in row]
                positions.append(torch.tensor(position))
            positions = torch.stack(positions)

        condition = torch.cat((actions, positions), axis=1)
        return condition

    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond = self.get_csv()
        return seq, cond
