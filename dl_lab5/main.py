import os
import sys
import random
import logging
from argparse import ArgumentParser

from typing import Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import save_model
from evaluator import evaluation_model
from iclevr_dataset import IclevrDataset
from models.dcgan import Discriminator as DC_Discriminator
from models.dcgan import Generator as DC_Generator
from models.acgan import Discriminator as AC_Discriminator
from models.acgan import Generator as AC_Generator

if not os.path.isdir("logs"):
    os.makedirs("logs")

log_filename = os.path.join("logs", "dl_lab5.log")
logging.basicConfig(filename=log_filename, level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger = logging.getLogger("dl_lab5")
logger.addHandler(handler)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = IclevrDataset("data", True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=16)
    test_dataset = IclevrDataset("data", False, "test.json")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=4)
    new_test_dataset = IclevrDataset("data", False, "new_test.json")
    new_test_loader = DataLoader(new_test_dataset, batch_size=len(new_test_dataset), num_workers=4)
    return train_loader, test_loader, new_test_loader


def train_dcgan(discriminator: nn.Module, generator: nn.Module, optimizer_d: optim,
                optimizer_g: optim, adversarial_loss: nn.modules.loss,
                train_loader: DataLoader, args: ArgumentParser):
    discriminator.train()
    generator.train()
    total_loss_d = 0
    total_loss_g = 0
    for images, conditions in train_loader:
        images = images.to(args.device)
        conditions = conditions.to(args.device)
        ############################
        # (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        optimizer_d.zero_grad()
        optimizer_g.zero_grad()
        # Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
        real_label = (0.3 * torch.rand(args.batch_size) + 0.7).to(args.device)
        # Use soft and noisy labels [0.0, 0.3]. Salimans et. al. 2016
        fake_label = (0.3 * torch.rand(args.batch_size) + 0.0).to(args.device)

        if random.random() < 0.05:
            real_label, fake_label = fake_label, real_label

        output = discriminator(images, conditions)
        loss_d_real = adversarial_loss(output, real_label)
        loss_d_real.backward()

        # train with fake
        noise = torch.randn(args.batch_size, args.latent_dim, 1, 1, device=args.device)
        fake_image = generator(noise, conditions)

        output = discriminator(fake_image.detach(), conditions)
        loss_d_fake = adversarial_loss(output, fake_label)
        loss_d_fake.backward()

        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        ############################
        # (2) Update Generator: maximize log(D(G(z)))
        ###########################
        generator_label = torch.ones(args.batch_size).to(args.device)
        output = discriminator(fake_image, conditions)
        loss_g = adversarial_loss(output, generator_label)
        loss_g.backward()
        optimizer_g.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
    return total_loss_d, total_loss_g, images, fake_image


def train_acgan(discriminator: nn.Module, generator: nn.Module, optimizer_d: optim, optimizer_g: optim,
                adversarial_loss: nn.modules.loss, auxiliary_loss: nn.modules.loss, train_loader: DataLoader,
                args: ArgumentParser):
    discriminator.train()
    generator.train()
    total_d_loss = 0
    total_g_loss = 0
    for images, conditions in train_loader:
        images = images.to(args.device)
        conditions = conditions.to(args.device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_d.zero_grad()
        optimizer_g.zero_grad()

        # Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
        real_label = (0.3 * torch.rand(args.batch_size) + 0.7).to(args.device)
        # Use soft and noisy labels [0.0, 0.3]. Salimans et. al. 2016
        fake_label = (0.3 * torch.rand(args.batch_size) + 0.0).to(args.device)

        if random.random() < 0.05:
            real_label, fake_label = fake_label, real_label

        # train with real
        validity, predicted_labels = discriminator(images)
        adv_loss = adversarial_loss(validity, real_label)
        aux_loss = auxiliary_loss(predicted_labels, conditions)
        d_real_loss = adv_loss + args.aux_weight * aux_loss
        d_real_loss.backward()

        # train with fake
        noise = torch.randn(args.batch_size, args.latent_dim, 1, 1, device=args.device)
        fake_images = generator(noise, conditions)
        validity, predicted_labels = discriminator(fake_images.detach())
        adv_loss = adversarial_loss(validity, fake_label)
        aux_loss = auxiliary_loss(predicted_labels, conditions)
        d_fake_loss = adv_loss + args.aux_weight * aux_loss
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        generator_label = torch.ones(args.batch_size).to(args.device)
        validity, predicted_labels = discriminator(fake_images)
        adv_loss = adversarial_loss(validity, generator_label)
        aux_loss = auxiliary_loss(predicted_labels, conditions)
        g_loss = adv_loss + args.aux_weight * aux_loss
        g_loss.backward()
        optimizer_g.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
    return total_d_loss, total_g_loss, images, fake_images


@torch.no_grad()
def evaluate(evaluator: nn.Module, generator: nn.Module, test_loader: DataLoader,
             args: ArgumentParser):
    generator.eval()
    acc = 0
    for _, conditions in test_loader:
        conditions = conditions.to(args.device)
        noise = torch.randn(len(test_loader.dataset), args.latent_dim, 1, 1, device=args.device)
        gen_image = generator(noise, conditions)
        acc = evaluator.eval(gen_image, conditions)
    return round(acc, 3), gen_image


def select_best_model(args):
    train_loader, test_loader, new_test_loader = load_data(args.batch_size)

    evaluator = evaluation_model(args.device, args.n_gpu)

    if args.is_dcgan:
        discriminator = DC_Discriminator(args.n_classes, args.ndf).to(args.device)
        generator = DC_Generator(args.n_classes, args.nc, args.latent_dim, args.ngf).to(args.device)
    else:
        discriminator = AC_Discriminator(args.n_classes, args.ndf).to(args.device)
        generator = AC_Generator(args.n_classes, args.nc, args.latent_dim, args.ngf).to(args.device)

    if args.n_gpu > 1:
        discriminator = nn.DataParallel(discriminator, list(range(args.n_gpu)))
        generator = nn.DataParallel(generator, list(range(args.n_gpu)))
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.BCELoss()

    # setup optimizer
    optimizer_d = torch.optim.SGD(discriminator.parameters(), lr=args.d_learning_rate, momentum=0.9)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.g_learning_rate, betas=(args.beta1, args.beta2))

    best_test_acc = 0
    best_new_test_acc = 0
    discriminator.train()
    generator.train()
    for epoch in range(1, args.n_epochs + 1):
        if args.is_dcgan:
            loss_d, loss_g, images, fake_image = train_dcgan(discriminator, generator, optimizer_d,
                                                             optimizer_g, adversarial_loss,
                                                             train_loader, args)
        else:
            loss_d, loss_g, images, fake_image = train_acgan(discriminator, generator, optimizer_d,
                                                             optimizer_g, adversarial_loss, auxiliary_loss,
                                                             train_loader, args)

        test_acc, test_image = evaluate(evaluator, generator, test_loader, args)
        new_test_acc, new_test_image = evaluate(evaluator, generator, new_test_loader, args)
        record = {'epoch': epoch, "loss_d": round(loss_d / len(train_loader), 5),
                  "loss_g": round(loss_g / len(train_loader), 5), "aux_weight": args.aux_weight,
                  "test_acc": test_acc, "new_test_acc": new_test_acc}
        logger.info(record)
        if test_acc + new_test_acc > best_test_acc + best_new_test_acc:
            best_test_acc = test_acc
            best_new_test_acc = new_test_acc
            save_model(args.model_folder, "discriminator", deepcopy(discriminator.state_dict()))
            save_model(args.model_folder, "generator", deepcopy(generator.state_dict()))

        if epoch % 5 == 0:
            save_image(test_image.detach(),
                       os.path.join(args.result_folder, "test_image_epoch{}.png".format(epoch)),
                       normalize=True)
    logger.info("Highest Test Accuracy: {}".format(best_test_acc))
    logger.info("Highest New Test Accuracy: {}".format(best_new_test_acc))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_epochs", default=600, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_classes", default=24, type=int)
    parser.add_argument("--nc", type=int, default=100, help="number of condition embedding dim")
    parser.add_argument("--latent_dim", type=int, default=100, help="size of the latent z vector")
    parser.add_argument("--ngf", type=int, default=300, help="size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="size of feature maps in discriminator")
    parser.add_argument("--g_learning_rate", default=0.0001, type=float)
    parser.add_argument("--d_learning_rate", default=0.0002, type=float)
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam. default=0.999")
    parser.add_argument('--result_folder', default="logs", help="folder to save result or log")
    parser.add_argument('--model_folder', default="logs/classifier", help="folder to save model")
    parser.add_argument('--is_dcgan', default=False)
    parser.add_argument('--aux_weight', default=125)
    parser.add_argument("--n_gpu", type=int, default=1, help='number of GPUs to use')
    parser.add_argument("--device", default="cuda:1", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)

    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    select_best_model(args)


if __name__ == '__main__':
    main()
