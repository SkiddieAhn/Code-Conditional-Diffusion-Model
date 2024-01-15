import torch
from fastprogress import progress_bar
import numpy as np

def beta_schedule_cosine(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def beta_schedule_linear(timesteps=1000):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.01
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    
def beta_schedule_quadratic(timesteps=1000):
    scale = 1000 / timesteps
    beta_start = (scale * 0.0001)**0.5
    beta_end = (scale * 0.01)**0.5
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)**2


def get_alpha_list(beta_list):
    return 1.0 - beta_list


def get_alpha_bar_list(alpha_list):
    return torch.cumprod(alpha_list, dim=0)


@torch.inference_mode()
def sampling(model, alpha_bar_list, condition=True, cfg_w=0.1, rsl=32, channels=3, num=10, total_timesteps=1000, sampling_timesteps=100, eta=0, device='cuda'):
    print(f"\n Sampling {num} new images....")

    if condition:
        c = [i for i in range(num)]
        c = np.array(c)
        c = torch.tensor(c, dtype=torch.long).to(device)
    else:
        c = None
        cfg_w = 0

    # get time_pairs 
    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    # get first image (noise)
    g_img = torch.randn((num, channels, rsl, rsl), dtype=torch.float32).to(device) 
    
    # Inference
    for time, time_next in progress_bar(time_pairs, total=len(time_pairs)):        
        in_t = torch.full((num,), time, dtype=torch.long).to(device)
        in_c = c

        # model forward with CFG
        pred_noise = model(g_img, in_t, in_c)
        if cfg_w > 0:
            uncond_pred_noise = model(g_img, in_t, None)
            pred_noise = (1+cfg_w)*pred_noise - cfg_w*uncond_pred_noise

        # set alpha
        alpha = alpha_bar_list[time]
        alpha_next = alpha_bar_list[time_next]
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_alpha_next = torch.sqrt(alpha_next)
    
        # predict x0
        numerator = g_img - (torch.sqrt(1-alpha) * pred_noise)
        pred_x0 = numerator / sqrt_alpha

        if time_next < 0:
            g_img = pred_x0
            
        else:        
            # predict direction_pointing 
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
            variance = sigma ** 2 
            direction_pointing = torch.sqrt(1-alpha_next-variance) * pred_noise

            # generate next image 
            g_img = (sqrt_alpha_next * pred_x0) + direction_pointing

    return g_img