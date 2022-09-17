import torch
from torch.optim import Adam

from model import UNet
from losses import p_losses
from diffusion import Scheduler, Diffuser


if __name__ == '__main__':
    global TIMESTEPS
    TIMESTEPS = 300

    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nodes = [500, 250, 100]
    model = UNet(nodes)
    optimiser = Adam(model.parameters(), lr = 1e-3)



    optimiser.zero_grad()
    batch = torch.randn((16,500))

    t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()

    scheduler = Scheduler(start = 0.0001, end = 0.02, timesteps = TIMESTEPS)
    diffuser = Diffuser(scheduler)

    loss = p_losses(model,diffuser, batch, t)
    loss.backward()
    optimiser.step()
