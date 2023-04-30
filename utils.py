import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
import math

import matplotlib.pyplot as plt

@torch.no_grad()
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_uniform_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.zeros_(m.bias.data)


def get_optimizer(model, lr: float = 1e-3, beta1: float = 0.5, beta2: float = 0.999):
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class ReceptiveFieldCalculator:
    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def calculate(self, input_image_size, depth, to_print: bool = False):
        n = input_image_size
        r = 1
        j = 1
        if to_print:
            print(f'initial: spatial size = {n}; receptive field = {r}')

        for i in range(depth):
            k = self.kernel_size
            s = self.stride
            p = self.padding

            n = math.floor((n - k + 2 * p) / s) + 1

            r = r + (k - 1) * j
            j = j * s
            if to_print:
                print(f'{i}: spatial size = {n}; receptive field = {r}')

        return r, n


def show_images(images, generated_images, masks):
    new_images = images * masks + generated_images * (~masks)
    for i, (img, new_image, generated_image, mask) in enumerate(zip(images, new_images, generated_images, masks)):
        if i > 11:
            break
        plt.figure()
        up = torch.cat([img, mask * img], dim=2)
        down = torch.cat([generated_image, new_image], dim=2)
        plt.imshow((torch.cat([up, down], dim=1).permute(1, 2, 0) + 1) / 2)
        plt.axis('off')


def extract_features(x, cnn):
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def gram_matrix(features, normalize=True):
    N, C, H, W = features.shape
    numel = H * W * C

    features = features.flatten(2, -1)
    gram = features.bmm(features.permute(0, 2, 1))
    if normalize:
        gram /= numel

    return gram


def sample_noise(imgs_size: torch.Tensor, dtype=torch.float32, device=torch.device('cpu')):
    return (2 * torch.rand(imgs_size) - 1).to(device).to(dtype)
