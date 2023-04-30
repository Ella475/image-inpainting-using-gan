import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
import torch.nn as nn
from options import Options as opt
from utils import gram_matrix

l1_loss = nn.L1Loss()


def _generator_least_squares_loss(scores_fake):
    loss = (scores_fake - 1).pow(2).mean() / 2
    return loss


def _generator_adversarial_loss(logits_fake, device):
    fake_label = torch.ones(logits_fake.size(), device=device)
    loss = bce_loss(logits_fake, fake_label)
    return loss


def _l1_loss(input_to_generator, output_to_generator):
    return l1_loss(input_to_generator, output_to_generator)


def _discriminator_least_squares_loss(scores_real, scores_fake):
    loss_real = (scores_real - 1).pow(2).mean()
    loss_fake = scores_fake.pow(2).mean()
    loss = (loss_real + loss_fake) / 2
    return loss, loss_real, loss_fake


def _discriminator_adversarial_loss(logits_real, logits_fake, device):
    real_label = torch.ones(logits_real.size(), device=device)
    fake_label = 1 - real_label
    loss_real = bce_loss(logits_real, real_label)
    loss_fake = bce_loss(logits_fake, fake_label)
    loss = loss_real + loss_fake
    return loss, loss_real, loss_fake


def style_loss(feats, style_layers, style_targets, style_weights):
    loss = 0.0
    for i in range(len(style_layers)):
        loss += style_weights[i] * (gram_matrix(feats[style_layers[i]]) - style_targets[i]).norm().pow(2)

    return loss


def tv_loss(img, tv_weight):
    img = img.squeeze()
    loss = tv_weight * ((img[:, :, 1:, :] - img[:, :, :-1, :]).norm().pow(2) + (img[:, :, :, 1:] - img[:, :, :, :-1]).norm().pow(2))
    return loss


