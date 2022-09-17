import torch
import torch.nn.functional as F

class Scheduler:

    def __init__(self, start, end, timesteps):
        self.start = start
        self.end = end
        self.timesteps = timesteps

        self.betas = self.linear_schedule()
        self.alpha = self.alphas()
        self.a_cp = self.alphas_cumprod()
        self.s_a_cp = self.sqrt_alphas_cumprod()
        self.s_r_a_cp = self.sqrt_recip_alphas()
        self.s_om_acp = self.sqrt_one_minus_alphas_cumprod()
        self.posteriors = self.posterior_variance()

    def linear_schedule(self):
        return torch.linspace(self.start, self.end, self.timesteps)

    def alphas(self):
        return 1-self.betas

    def alphas_cumprod(self):
        return torch.cumprod(self.alpha, axis=0)

    def sqrt_alphas_cumprod(self):
        return torch.sqrt(self.a_cp)

    def sqrt_recip_alphas(self):
        return torch.sqrt(1/self.alpha)

    def sqrt_one_minus_alphas_cumprod(self):
        return torch.sqrt(1-self.a_cp)

    def posterior_variance(self):
        alphas_cumprod_prev = F.pad(self.a_cp[:-1], (1, 0), value=1.0)
        return self.betas * (1-alphas_cumprod_prev) / (1. -self.a_cp)



def get_noise_from_index(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffuser:

    def __init__(self, scheduler):
        self.scheduler = scheduler


    def diffuse(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = get_noise_from_index(self.scheduler.s_a_cp, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_noise_from_index(self.scheduler.s_om_acp, t, x.shape)

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
