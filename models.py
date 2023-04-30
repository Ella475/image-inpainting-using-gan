import torch
import torch.nn as nn
from torchsummary import summary
from options import Options
from utils import ReceptiveFieldCalculator, Flatten, Concatenate

gbnc = Options.generator_base_num_channels
dbnc = Options.discriminator_base_num_channels


class Identity(nn.Module):
    def forward(self, x):
        return x


class EncoderBlock(nn.Module):
    def __init__(self, nc: int, input_nc: int, first: bool = False, last: bool = False, last_channels: int = None,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        '''
        First: conv
        Inner: relu, conv, norm
        Last: relu, conv
        '''
        super(EncoderBlock, self).__init__()
        model_list = [nn.LeakyReLU(0.2, True) if not first else Identity(),
                      nn.Conv2d(nc if not first else input_nc,
                                last_channels if last_channels is not None and last else 2 * nc,
                                kernel_size=kernel_size, stride=stride, padding=padding),
                      nn.BatchNorm2d(2 * nc) if not first and not last else Identity()]

        self.model = nn.Sequential(*model_list)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, nc: int, output_nc: int, first: bool = False, last: bool = False, u_net: bool = True):
        super(DecoderBlock, self).__init__()
        model_list = [nn.ReLU(True),
                      nn.ConvTranspose2d((nc * 2) if u_net and not first else nc, (nc // 2) if not last else output_nc,
                                         kernel_size=4,
                                         stride=2, padding=1),
                      nn.BatchNorm2d(nc // 2) if not last else Identity(),
                      nn.Tanh() if last else Identity()]

        self.model = nn.Sequential(*model_list)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class ConnectionBlock(nn.Module):
    def __init__(self, nc: int, depth: int = 1):
        super(ConnectionBlock, self).__init__()
        model_list = [nn.Sequential(nn.ReLU(True),
                                    nn.Conv2d(nc, nc, kernel_size=1))
                      for _ in range(depth)]

        self.model = nn.Sequential(*model_list)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class GeneratorBlock(nn.Module):
    def __init__(self, down_block: EncoderBlock, up_block: DecoderBlock, inner_block=None,
                 u_net: bool = True, first: bool = False, ):
        super(GeneratorBlock, self).__init__()
        model_list = [down_block,
                      inner_block if inner_block is not None else Identity(),
                      up_block]

        self.model = nn.Sequential(*model_list)
        self.u_net = u_net
        self.first = first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.u_net and not self.first:
            # check that u-net calculation is correct
            assert x.size()[1] == out.size()[1]

            return torch.cat([x, out], 1)
        else:
            return out


class Generator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 3, levels: int = 7, u_net: bool = True):
        super(Generator, self).__init__()

        nc = gbnc * (2 ** levels)
        block = ConnectionBlock(nc, 3)
        for i in range(levels - 1, -1, -1):
            nc //= 2
            encoder_first = True if i == 0 else False
            encoder_last = True if i == levels - 1 else False
            decoder_first = encoder_last
            decoder_last = encoder_first

            down = EncoderBlock(nc=nc, input_nc=input_nc, first=encoder_first, last=encoder_last)
            up = DecoderBlock(nc=2 * nc, output_nc=output_nc, first=decoder_first, last=decoder_last, u_net=u_net)
            block = GeneratorBlock(down, up, inner_block=block, u_net=u_net, first=encoder_first)

        self.model = block

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, base_nc: int = 4, output_size: int = 1024,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1, levels: int = 5, activation_size: int = 4,
                 remove_head: bool = False, last_channels: int = None):
        super(PatchDiscriminator, self).__init__()

        nc = base_nc
        model_list = []
        for i in range(0, levels):
            nc *= 2
            encoder_first = True if i == 0 else False
            encoder_last = True if i == levels - 1 else False
            model_list.append(EncoderBlock(nc=nc, input_nc=input_nc, first=encoder_first, last=encoder_last,
                                           last_channels=last_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        if not remove_head:
            model_list.extend(
                [Flatten(), nn.LeakyReLU(0.2, True), nn.Linear((activation_size ** 2) * 2 * nc, output_size)])
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, patch_size: int = 40, input_size: int = 128, input_nc: int = 3, output_size: int = 128,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super(Discriminator, self).__init__()

        depth_gd, activation_size_gd = self._compute_depth(input_size, input_size, kernel_size, stride, padding)
        depth_ld, activation_size_ld = self._compute_depth(input_size, patch_size, kernel_size, stride, padding)

        self.base_net = PatchDiscriminator(input_nc=input_nc, base_nc=dbnc,
                                           kernel_size=kernel_size, stride=stride, padding=padding,
                                           levels=depth_ld, remove_head=True)
        self.gd = PatchDiscriminator(input_nc=dbnc * 2 ** (depth_ld+1), base_nc=dbnc * 2 ** depth_ld, output_size=1,
                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                     levels=depth_gd - depth_ld, activation_size=activation_size_gd)
        self.ld = PatchDiscriminator(input_nc=dbnc * 2 ** (depth_ld+1), base_nc=dbnc * 4,
                                     kernel_size=1, stride=1, padding=0, last_channels=1,
                                     levels=2, activation_size=activation_size_ld, remove_head=True)
        self.concat = Concatenate()

    def forward(self, x):
        x_base = self.base_net(x)
        x_ld = self.ld(x_base)  # (N, 1, H, W)
        x_gd = self.gd(x_base)  # (N, 1, 1, 1)
        return self.concat([x_gd, x_ld.view(x_ld.shape[0], -1)])

    def _compute_depth(self, input_size: int, patch_size: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        calculator = ReceptiveFieldCalculator(kernel_size=kernel_size, stride=stride, padding=padding)

        depth = 0
        r = 1
        n = input_size
        while r < patch_size:
            depth += 1
            r, n = calculator.calculate(input_image_size=input_size, depth=depth)

        return depth, n


if __name__ == '__main__':
    levels = 5
    input_nc_g = 4
    g = Generator(input_nc=input_nc_g, output_nc=3, levels=levels, u_net=True)
    data = torch.randn((10, input_nc_g, Options.input_size, Options.input_size))
    with torch.no_grad():
        data = g.forward(data)
    summary(g, (input_nc_g, Options.input_size, Options.input_size), depth=20)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    d = Discriminator(input_size=Options.input_size)
    summary(d, (3, Options.input_size, Options.input_size), depth=30)

    calculator = ReceptiveFieldCalculator(kernel_size=4, stride=2, padding=1)
    calculator.calculate(input_image_size=Options.input_size, depth=3, to_print=True)
