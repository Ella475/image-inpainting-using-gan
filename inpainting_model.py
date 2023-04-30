import torch
import torchvision
import torchvision.transforms as transforms

from models import Generator, Discriminator
from loss_functions import _l1_loss, _generator_adversarial_loss, _generator_least_squares_loss
from loss_functions import _discriminator_adversarial_loss, _discriminator_least_squares_loss, tv_loss, style_loss
from options import Options as opt
from utils import get_optimizer, extract_features, gram_matrix, sample_noise


class InpaintingModel:
    def __init__(self, is_train: bool = True, device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        self.isTrain = is_train
        self.device = device
        self.dtype = dtype
        self.input = None
        self.masks = None
        self.masked_input = None
        self.masked_input_with_mask = None
        self.generated_images = None
        self.loss_D = None
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_G = None
        self.loss_style = None
        self.loss_tv = None
        self.net_G = Generator(input_nc=4, output_nc=3, levels=5, u_net=True).to(self.device)

        if self.isTrain:
            self.net_D = Discriminator(input_nc=3).to(self.device).to(self.dtype)
            self.optimizer_G = get_optimizer(model=self.net_G, lr=opt.g_lr, beta1=opt.g_beta1)
            self.optimizer_D = get_optimizer(model=self.net_D, lr=opt.d_lr, beta1=opt.d_beta1)

            if opt.use_style:
                self.style_cnn = torchvision.models.vgg13_bn(pretrained=True).features
                self.style_cnn.to(self.dtype).to(self.device)
                self._set_requires_grad(self.style_cnn, True)

    def set_input(self, input: torch.Tensor, masks: torch.Tensor):
        self.masks = masks.to(self.device)
        self.input = input.to(self.device).to(self.dtype)
        self.masked_input = self.masks * self.input + (~self.masks) * sample_noise(self.input.size(),
                                                                                                  self.dtype, self.device)
        self.masked_input_with_mask = torch.cat([self.masked_input, self.masks], dim=1)

    def forward(self):
        self.generated_images = self.net_G(self.masked_input_with_mask)  # G(A)
        return self.generated_images

    def backward_D(self):
        pred_fake = self.net_D(self.generated_images.detach())
        pred_real = self.net_D(self.input)
        self.loss_D, self.loss_D_real, self.loss_D_fake = self.discriminator_loss(pred_real, pred_fake)
        self.loss_D.backward()

    def backward_G(self):
        pred_fake = self.net_D(self.generated_images)
        self.loss_G = self.generator_loss(pred_fake, self.input, self.generated_images)
        self.loss_G.backward()

    def compute_style_loss(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        input = normalize((self.input + 1) / 2)
        feats = extract_features(input, self.style_cnn)
        style_targets = []
        for idx in opt.style_layers:
            style_targets.append(gram_matrix(feats[idx].clone()))
        generated = normalize((self.generated_images + 1) / 2)
        feats = extract_features(generated, self.style_cnn)
        s_loss = style_loss(feats, opt.style_layers, style_targets, opt.style_weights)
        t_loss = tv_loss(self.generated_images, opt.tv_weight)
        self.loss_style = s_loss
        self.loss_tv = t_loss
        return self.loss_style + self.loss_tv

    def generator_loss(self, scores_fake, input_to_generator, output_to_generator):
        if opt.ls_gan_mode:
            gan_loss = _generator_least_squares_loss(scores_fake)
        else:
            gan_loss = _generator_adversarial_loss(scores_fake, self.device)

        loss = (1 - opt.l1_weight) * gan_loss + opt.l1_weight * _l1_loss(input_to_generator, output_to_generator)
        if opt.use_style:
            loss += self.compute_style_loss()
        return loss

    def discriminator_loss(self, scores_real, scores_fake):
        if opt.ls_gan_mode:
            return _discriminator_least_squares_loss(scores_real, scores_fake)
        return _discriminator_adversarial_loss(scores_real, scores_fake, self.device)

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self._set_requires_grad(self.net_D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self._set_requires_grad(self.net_D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    @staticmethod
    def _set_requires_grad(net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def eval(self):
        self.net_G.eval()
        if self.isTrain:
            self.net_D.eval()
            self.style_cnn.eval()

    def train(self):
        self.net_G.train()
        if self.isTrain:
            self.net_D.train()
            self.style_cnn.train()

