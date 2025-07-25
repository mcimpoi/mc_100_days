import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from datasets import load_dataset

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
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    x_noisy = q_sample(x_start, t)
    noisy_image = reverse_tensor_transform(x_noisy.squeeze())
    return noisy_image


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(predicted_noise, noise)
    elif loss_type == "l2":
        loss = F.mse_loss(predicted_noise, noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(predicted_noise, noise)
    else:
        raise NotImplementedError(f"Unknown loss type: {loss_type}")

    return loss


def get_fashion_mnist_dataloader(batch_size: int = 128) -> torch.utils.data.DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # Normalize to
        ]
    )

    def transforms_fn(examples: dict) -> dict:
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    dataset = load_dataset("fashion_mnist")
    transformed_dataset = dataset.with_transform(transforms_fn).remove_columns("label")
    dataloader = torch.utils.data.DataLoader(
        transformed_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader
    
