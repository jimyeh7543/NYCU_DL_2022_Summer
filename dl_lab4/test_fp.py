import torch
import argparse

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import bair_robot_pushing_dataset
from utils import pred, finn_eval_seq, save_gif_with_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--data_root', default='data/processed_data', help='root directory for data')
    parser.add_argument('--model_path', default='saved_model', help='path to model')
    parser.add_argument('--gif_dir', default='plot_pred_images', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')

    args = parser.parse_args()
    return args


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w + 2 * pad + 30, w + 2 * pad))
    if color == 'red':
        px[0] = 0.7
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w + pad, pad:w + pad] = x
    else:
        px[:, pad:w + pad, pad:w + pad] = x
    return px


def make_gifs(x, cond, modules, args, device, idx, name):
    # get approx posterior sample
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    posterior_gen = [x[0]]
    x_in = x[0]
    for i in range(1, args.n_past + args.n_future):
        h = modules['encoder'](x_in)
        h_target = modules['encoder'](x[i])[0].detach()
        if args.last_frame_skip or i < args.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _ = modules['posterior'](h_target)  # take the mean
        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1))
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1)).detach()
            x_in = modules['decoder']([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    all_gen = []
    gen_seq = []
    gt_seq = []
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    x_in = x[0]
    all_gen.append(x_in)
    for i in range(1, args.n_past + args.n_future):
        h = modules['encoder'](x_in)
        if args.last_frame_skip or i < args.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()

        if i < args.n_past:
            h_target = modules['encoder'](x[i])[0].detach()
            z_t, _, _ = modules['posterior'](h_target)
            modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1))
            x_in = x[i]
            all_gen.append(x_in)
        else:
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            h = modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in)
            gt_seq.append(x[i])
            all_gen.append(x_in)
    _, _, psnr = finn_eval_seq(gt_seq, gen_seq)

    ###### psnr ######
    gifs = [[] for t in range(args.n_past + args.n_future)]
    text = [[] for t in range(args.n_past + args.n_future)]
    mean_psnr = np.mean(psnr, 1)
    ordered = np.argsort(mean_psnr)
    best_psnr_index = ordered[-1]
    rand_sidx = [np.random.randint(args.batch_size) for s in range(3)]
    for t in range(args.n_past + args.n_future):
        # gt
        gifs[t].append(add_border(x[t][best_psnr_index], 'green'))
        text[t].append('Ground\ntruth')
        # posterior
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(posterior_gen[t][best_psnr_index], color))
        text[t].append('Approx.\nposterior')
        # best
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(all_gen[t][best_psnr_index], color))
        text[t].append('Best PSNR')
        # random 3
        for s in range(len(rand_sidx)):
            gifs[t].append(add_border(all_gen[t][rand_sidx[s]], color))
            text[t].append('Random\nsample %d' % (s + 1))

    filename = '%s/%s_%d.gif' % (args.gif_dir, name, idx)
    save_gif_with_text(filename, gifs, text)


def main():
    args = parse_args()
    device = "cuda:5"

    # Load model
    saved_model = torch.load('%s/model.pth' % args.model_path)
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']

    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }

    # To gpu
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # Set the args
    args.last_frame_skip = saved_model['args'].last_frame_skip
    args.z_dim = saved_model['args'].z_dim
    print(args)

    # Load test dataset
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    generate_git = True
    if generate_git:
        print("Start to generate gif")
        for test_seq, test_cond in test_loader:
            test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
            test_cond = test_cond.permute(1, 0, 2).to(device)
            make_gifs(test_seq, test_cond, modules, args, device, 0, "test")
            break

    test_all = True
    if test_all:
        psnr_list = []
        for test_seq, test_cond in test_loader:
            test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
            test_cond = test_cond.permute(1, 0, 2).to(device)
            pred_seq = pred(test_seq, test_cond, modules, args, device)
            _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
            seq_psnr = np.mean(psnr, 1)
            psnr_list.extend(seq_psnr)

        max_psnr = np.max(psnr_list)
        min_psnr = np.min(psnr_list)
        avg_psnr = np.mean(psnr_list)
        print("Max PSNR: {}".format(max_psnr))
        print("Min PSNR: {}".format(min_psnr))
        print("Avg PSNR on test dataset: {}".format(avg_psnr))


if __name__ == '__main__':
    main()
