import math
import os
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
tensor_to_pil_image = transforms.ToPILImage()


def kl_criterion(mu, logvar, args):
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size
    return KLD

def kl_criterion_lp(mu1, logvar1, mu2, logvar2, args):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(
    #
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / args.batch_size

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c])
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err


# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr


def finn_psnr(x, y, data_range=1.):
    mse = ((x - y) ** 2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def plot_pred(seq, cond, modules, args, epoch, device):
    if not os.path.exists("plot_pred_images"):
        os.makedirs("plot_pred_images")
    gen_seq = pred(seq, cond, modules, args, device)
    new_image = Image.new(mode="RGB", size=(gen_seq[0].shape[3] * len(gen_seq), gen_seq[0].shape[2] * 2))
    for i in range(len(gen_seq)):
        ori_img = tensor_to_pil_image(seq[i][0])
        gen_img = tensor_to_pil_image(gen_seq[i][0])
        new_image.paste(ori_img, (i * ori_img.size[0], 0))
        new_image.paste(gen_img, (i * ori_img.size[0], ori_img.size[1]))
    new_image.save(os.path.join("plot_pred_images", "{}.jpg".format(epoch)))

def plot_pred_lp(seq, cond, modules, args, epoch, device):
    if not os.path.exists("plot_pred_images"):
        os.makedirs("plot_pred_images")
    gen_seq = pred_lp(seq, cond, modules, args, device)
    new_image = Image.new(mode="RGB", size=(gen_seq[0].shape[3] * len(gen_seq), gen_seq[0].shape[2] * 2))
    for i in range(len(gen_seq)):
        ori_img = tensor_to_pil_image(seq[i][0])
        gen_img = tensor_to_pil_image(gen_seq[i][0])
        new_image.paste(ori_img, (i * ori_img.size[0], 0))
        new_image.paste(gen_img, (i * ori_img.size[0], ori_img.size[1]))
    new_image.save(os.path.join("plot_pred_images", "{}.jpg".format(epoch)))


def pred(x, cond, modules, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    gen_seq = [x[0]]
    x_in = x[0]
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
            gen_seq.append(x[i])
        else:
            z_t = torch.randn(args.batch_size, args.z_dim).to(device)
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1)).detach()
            x_in = modules['decoder']([h_pred, skip]).detach()
            gen_seq.append(x_in)
    return gen_seq

def pred_lp(x, cond, modules, args, device):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    modules['prior'].hidden = modules['prior'].init_hidden()
    gen_seq = [x[0]]
    x_in = x[0]
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
            modules['prior'](h)
            modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1))
            x_in = x[i]
            gen_seq.append(x[i])
        else:
            z_t, _, _ = modules['prior'](h)
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, cond[i - 1]], 1)).detach()
            x_in = modules['decoder']([h_pred, skip]).detach()
            gen_seq.append(x_in)
    return gen_seq


def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x * 255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0, 0, 0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images) - 1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding:
                      (i + 1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding:
                         (i + 1) * y_dim + i * padding].copy_(image)
        return result

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
             hasattr(arg, "__iter__")))
