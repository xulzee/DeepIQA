import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import conv2d, conv_transpose2d

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
kern = Variable(torch.FloatTensor(k5x5).cuda())


def downsample_img(img):
    kernel = kern
    return conv2d(input=img, weight=kernel, padding=2, stride=2)


def upsample_img(img):
    kernel = kern * 4
    return conv_transpose2d(input=img, weight=kernel, padding=2, stride=2, output_padding=1)


def normalize_lowpass_subt(img, n_level=3):
    '''Normalize image by subtracting the low-pass-filtered image'''
    # Downsample
    img_ = img
    for i in range(n_level - 1):
        img_ = downsample_img(img_)
    # Upsample
    for i in range(n_level - 1):
        img_ = upsample_img(img_)
    return img - img_


def log_diff_fn(in_a, in_b, eps=0.1):
    diff = 255.0 * (in_a - in_b)
    log_255_sq = (2 * torch.log(torch.FloatTensor([255.0])))
    val = log_255_sq - torch.log(diff ** 2 + eps)
    max_val = (log_255_sq - torch.log(torch.FloatTensor([eps])))
    return val / max_val
