import os
import random

import torch
import torchvision.transforms as T
from PIL import Image


def center_crop_mask(image_size: int = 128, crop_factor: int = 4, device: torch.device = torch.device('cpu')):
    img = torch.ones((3, image_size, image_size), dtype=torch.bool, device=device)
    crop_size = image_size // crop_factor
    img[:, crop_size:-crop_size, crop_size:-crop_size] = 0
    return img


def random_blocks_mask(image_size: int = 128, crop_factor: int = 6, max_num_crops: int = 6,
                       device: torch.device = torch.device('cpu')):
    img = torch.ones((3, image_size, image_size), dtype=torch.bool, device=device)
    crop_size = image_size // crop_factor
    crop_limit = int(min((image_size ** 2 / 4) // (crop_size ** 2), max_num_crops))
    for _ in range(crop_limit):
        x = random.randint(1, image_size - crop_size)
        y = random.randint(1, image_size - crop_size)
        img[:, y:y + crop_size, x:x + crop_size] = 0

    return img


def random_region_mask(masks_path: str = 'segmentation', image_size: int = 128, crop_factor: int = 2,
                       device: torch.device = torch.device('cpu')):
    img = torch.ones((3, image_size, image_size), dtype=torch.bool, device=device)
    crop_size = image_size // crop_factor
    x = random.randint(1, image_size - crop_size)
    y = random.randint(1, image_size - crop_size)

    random_path = random.choice([
        os.path.join(masks_path, x) for x in os.listdir(masks_path)
        if os.path.isfile(os.path.join(masks_path, x))
    ])

    with Image.open(random_path) as I:
        mask = I.resize((crop_size, crop_size))
    mask = mask.convert("RGB")
    t = T.ToTensor()
    mask = t(mask).to(device)
    mask = (255 * mask).sum(dim=0).to(torch.bool)
    img[:, y:y + crop_size, x:x + crop_size] = ~mask
    return img


def random_crop_generator(image_size: int = 128, probs=None, device: torch.device = torch.device('cpu')):
    if probs is None:
        probs = [0.01, 0.454, 0.455]
    n = random.random()
    if n < probs[0]:
        return center_crop_mask(image_size=image_size, device=device)
    elif n < probs[1]:
        return random_region_mask(image_size=image_size, device=device)
    else:
        return random_blocks_mask(image_size=image_size, device=device)


def random_batch_masks(image_size: int = 128, batch_size=120, squeeze: bool = True,
                       device: torch.device = torch.device('cpu'), probs=None):
    masks = [random_crop_generator(image_size=image_size, device=device, probs=probs) for _ in range(batch_size)]
    if squeeze:
        return torch.stack(masks, dim=0)[:, 0, :, :].unsqueeze(dim=1)
    else:
        return torch.stack(masks, dim=0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    masks = random_batch_masks(batch_size=5, squeeze=False)

    for i in range(5):
        mask = masks[i]
        plt.figure()
        plt.imshow(255 * mask.permute(1, 2, 0))
        plt.axis('off')

    plt.show()
