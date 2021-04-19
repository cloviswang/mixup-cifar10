import random

import numpy as np
import torch
import torchvision
from PIL import Image

import utils
from data_augmentation import augmentations


def augments(input):
    output = np.transpose(input.cpu().numpy(), (0, 2, 3, 1))
    for i in range(0, 127):
        output[i] = augment_and_mix(output[i])
    output = np.transpose(output, (0, 3, 1, 2))
    output = torch.from_numpy(output).cuda()
    return output


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix
    return mixed


def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, args, dataset, preprocess, no_jsd=True):
        self.args = args
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), self. aug(x, self.preprocess),
                        self.aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)

    def aug(self, image, preprocess):
        """Perform AugMix augmentations and compute mixture.

        Args:
          image: PIL.Image input image
          preprocess: Preprocessing function which should return a torch tensor.

        Returns:
          mixed: Augmented and mixed image.
        """
        aug_list = augmentations.augmentations
        if self.args.all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * self.args.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(self.args.mixture_width):
            image_aug = image.copy()
            depth = self.args.mixture_depth if self.args.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, self.args.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed
