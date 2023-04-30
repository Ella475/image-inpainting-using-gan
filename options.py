from dataclasses import dataclass


@dataclass
class Options:
    batch_size = 100
    input_size = 256
    dataset_root = 'dataset'
    workers = 4
    generator_base_num_channels = 16
    discriminator_base_num_channels = 4
    ls_gan_mode = True
    l1_weight = 0.9
    use_style = True
    style_weights = 5 * (0.5, 1.25e-3, 3e-5, 2.5e-6)
    style_layers = (1, 4, 6, 7)
    tv_weight = 5e-7
    g_lr = 0.0002
    g_beta1 = 0.5
    d_lr = 0.0002
    d_beta1 = 0.5
