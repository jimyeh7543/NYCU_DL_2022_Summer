import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, n_classes, nc, nz, ngf):
        super(Generator, self).__init__()
        self.nc = nc

        self.condition_emb = nn.Sequential(
            nn.Linear(n_classes, self.nc),
            nn.LeakyReLU(0.2, True)
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + self.nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (RGB channel=3) x 64 x 64
        )

    def forward(self, input_data, conditions):
        condition_emb = self.condition_emb(conditions).view(-1, self.nc, 1, 1)
        input_with_condition = torch.cat((condition_emb, input_data), 1)
        output = self.main(input_with_condition)
        return output


class Discriminator(nn.Module):
    def __init__(self, n_classes, ndf):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.main = nn.Sequential(
            # input is 3 (RGB channel) x 64 x 64
            nn.Conv2d(3, self.ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, self.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(32 * self.ndf * 8 * 8, 1),
            nn.Sigmoid()
        )

        self.fc_aux = nn.Sequential(
            nn.Linear(32 * self.ndf * 8 * 8, n_classes),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        output = self.main(input_data)
        output = output.view(output.shape[0], -1)
        validity = self.fc_adv(output).squeeze(1)
        predicted_labels = self.fc_aux(output)
        return validity, predicted_labels
