import numpy as np
import torch
from kornia.filters import MedianBlur as korniaMedianBlur

def gaussian_kernel_2d(kernel_size, sigma):
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d

class GaussianNoise():
    def __init__(self, std):
        self.std = std
    def __call__(self, image):
        return image + self.std * torch.randn_like(image)

class GaussianBlur():
    def __init__(self, kernel_size, sigma):
        self.kernel = gaussian_kernel_2d(kernel_size, sigma)[None, None, :].to('cuda')
        self.padding = (kernel_size-1)//2
    def __call__(self, image):
        C = image.shape[1]
        return torch.nn.functional.conv2d(image, self.kernel.repeat(C,1,1,1), padding=self.padding, groups=C)

class MedianBlur(korniaMedianBlur):
    def __init__(self, kernel_size):
        super().__init__((kernel_size, kernel_size))

class SaltPepper():
    def __init__(self, prob):
        self.sampler = torch.distributions.Bernoulli(prob) # chance 1
        self.flip = torch.distributions.Bernoulli(0.5)
    def __call__(self, image):
        positive = self.sampler.sample(image.shape).to(image.device) * 2
        flip = self.flip.sample(image.shape).to(image.device) * 2 - 1
        return (image + flip * positive).clamp(-1,1)

def Noiser(image):
        n = np.random.choice([GaussianBlur(7, 2), GaussianBlur(5, 2), GaussianBlur(3, 2), GaussianNoise(0.1), SaltPepper(0.05), MedianBlur(7), MedianBlur(5), MedianBlur(3), FJPEG()], p=[1/15,1/15,1/15,1/5,1/5,1/15,1/15,1/15,1/5])
        return n(image.clamp(-1,1)).clamp(-1,1)
