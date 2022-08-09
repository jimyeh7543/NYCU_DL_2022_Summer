import os
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from models.lstm import gaussian_lstm, lstm
from dataset import bair_robot_pushing_dataset
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion_lp, finn_eval_seq, pred_lp, plot_pred_lp

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/lp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--n_epochs', type=int, default=201)
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=15,
                        help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0,
                        help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0,
                        help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3,
                        help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--n_cond', type=int, default=7)
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true',
                        help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')

    args = parser.parse_args()
    args = initialize_tfr_decay_step(args)
    return args


def initialize_tfr_decay_step(args):
    args.tfr_decay_step = (1.0 - args.tfr_lower_bound) / (args.n_epochs - args.tfr_start_decay_epoch)
    return args


def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['prior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    mse_criterion = nn.MSELoss()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    modules['prior'].hidden = modules['prior'].init_hidden()
    mse = 0
    kld = 0
    h_seq = [0] * (args.n_past + args.n_future)
    for i in range(0, args.n_past + args.n_future):
        h_seq[i] = modules['encoder'](x[i])
    use_teacher_forcing = True if random.random() < args.tfr else False
    for i in range(1, args.n_past + args.n_future):
        if args.last_frame_skip or i < args.n_past:
            h, skip = h_seq[i - 1]
        else:
            h = h_seq[i - 1][0]

        z_t, mu, logvar = modules['posterior'](h_seq[i][0])
        _, mu_p, logvar_p = modules['prior'](h)
        h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1))
        x_pred = modules['decoder']([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion_lp(mu, logvar, mu_p, logvar_p, args)
        if not use_teacher_forcing:
            h_seq[i] = modules['encoder'](x_pred)
        # raise NotImplementedError

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (
            args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)


class KlAnnealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical = args.kl_anneal_cyclical
        self.ratio = args.kl_anneal_ratio
        self.n_cycles = args.kl_anneal_cycle
        self.n_epochs = args.n_epochs

        self.period = self.n_epochs / self.n_cycles
        if not self.cyclical:
            self.ratio = 1
        self.step = 1 / (self.period * self.ratio)

        self.index = 0
        self.cycle = 0
        self.beta = 0
        # raise NotImplementedError

    def update(self):
        self.index += 1
        if self.cyclical:
            if self.index >= (self.cycle + 1) * self.period:
                self.cycle += 1
            self.beta = (self.index - (self.cycle * self.period)) * self.step
        else:
            self.beta = self.index * self.step

        if self.beta > 1:
            self.beta = 1
        # raise NotImplementedError

    def get_beta(self):
        return self.beta


def main():
    args = parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda:5'
    else:
        device = 'cpu'

    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        n_epochs = args.n_epochs
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f' \
               % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.prior_rnn_layers,
                  args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        n_epochs = args.n_epochs
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))

    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        prior = saved_model['prior']
    else:
        frame_predictor = lstm(args.g_dim + args.z_dim + args.n_cond, args.g_dim, args.rnn_size,
                               args.predictor_rnn_layers,
                               args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size,
                                  device)
        prior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.prior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
        prior.apply(init_weights)

    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    prior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    test_data = bair_robot_pushing_dataset(args, 'test')
    train_loader = DataLoader(train_data,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    validate_loader = DataLoader(validate_data,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True)

    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(prior.parameters()) + list(
        encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = KlAnnealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.n_epochs)
    best_test_psnr = 0
    for epoch in range(start_epoch, start_epoch + n_epochs):
        frame_predictor.train()
        posterior.train()
        prior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for seq, cond in train_loader:
            seq = seq.permute(1, 0, 2, 3, 4).to(device)
            cond = cond.permute(1, 0, 2).to(device)
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld

        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            args.tfr -= args.tfr_decay_step
            # raise NotImplementedError

        progress.update(1)
        record = {'epoch': epoch, 'tfr': args.tfr, "loss": round(epoch_loss / len(train_loader), 5),
                  "mse_loss": round(epoch_mse / len(train_loader), 5),
                  "kld_loss": round(epoch_kld / len(train_loader), 5),
                  "kl_weight": kl_anneal.get_beta()}
        with open(os.path.join(args.log_dir, "train_record.txt"), 'a') as train_record:
            train_record.write(json.dumps(record) + "\n")
        kl_anneal.update()

        frame_predictor.eval()
        posterior.eval()
        prior.eval()
        encoder.eval()
        decoder.eval()

        if epoch % 2 == 0:
            psnr_list = []
            for validate_seq, validate_cond in validate_loader:
                validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
                validate_cond = validate_cond.permute(1, 0, 2).to(device)
                pred_seq = pred_lp(validate_seq, validate_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)

            avg_psnr = np.mean(np.concatenate(psnr_list))
            record = {'epoch': epoch, 'validation_psnr': round(avg_psnr, 5)}
            with open(os.path.join(args.log_dir, "train_record.txt"), 'a') as train_record:
                train_record.write(json.dumps(record) + "\n")

            psnr_list = []
            for test_seq, test_cond in test_loader:
                test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
                test_cond = test_cond.permute(1, 0, 2).to(device)
                pred_seq = pred_lp(test_seq, test_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)

            avg_psnr = np.mean(np.concatenate(psnr_list))
            record = {'epoch': epoch, 'test_psnr': round(avg_psnr, 5)}
            with open(os.path.join(args.log_dir, "train_record.txt"), 'a') as train_record:
                train_record.write(json.dumps(record) + "\n")

            if avg_psnr > best_test_psnr:
                best_test_psnr = avg_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'prior': prior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        if epoch % 20 == 0:
            for validate_seq, validate_cond in validate_loader:
                validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
                validate_cond = validate_cond.permute(1, 0, 2).to(device)
                plot_pred_lp(validate_seq, validate_cond, modules, args, epoch, device)
                break


if __name__ == '__main__':
    main()
