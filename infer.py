import torch
from tqdm import tqdm
from diffusion import get_noise_from_index

@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index):
    betas_t = get_noise_from_index(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_noise_from_index(
        scheduler.s_om_acp, t, x.shape
    )
    sqrt_recip_alphas_t = get_noise_from_index(scheduler.s_r_a_cp, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = get_noise_from_index(scheduler.posteriors, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    x = torch.randn(shape, device=device)
    xs = []

    for i in tqdm(reversed(range(0, TIMESTEPS)), desc='sampling loop time step', total=TIMESTEPS):
        x = p_sample(model, x, torch.full((b,), i, device=device, dtype=torch.long), i)
        xs.append(x.cpu().numpy())
    return xs


@torch.no_grad()
def sample(model, size, batch_size=16, channels=1):
    return p_sample_loop(model, shape=(batch_size, channels, size, size))
