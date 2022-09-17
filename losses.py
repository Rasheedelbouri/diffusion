import torch
from torch import nn


LOSS = nn.L1Loss()

def p_losses(model, diffuser, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = diffuser.diffuse(x=x_start, t=t)
    predicted_noise = model.forward(x_noisy, t)
    loss = LOSS(noise, predicted_noise)

    return loss
