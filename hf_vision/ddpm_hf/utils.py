import torch
import numpy as np
from torchvision import transforms

IMAGE_SIZE: int = 128
N_TIMESTEPS: int = 300

to_tensor_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # Normalize to [-1, 1]
    ]
)

reverse_tensor_transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: (x + 1) / 2),  # Normalize to [0, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda x: x * 255.0),
        transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ]
)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def extract(a, t, x_shape):
    batch_size = t.shape[0]

    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    betas = linear_beta_schedule(N_TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)  
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise