import torch
import configparser

from src.MCMC_Sampling.grad_log_probs import prior_grad_log, posterior_grad_log

torch.autograd.set_detect_anomaly(True)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])

prior_steps = int(parser["MCMC"]["E_SAMPLE_STEPS"])
prior_s = torch.tensor(float(parser["MCMC"]["E_STEP_SIZE"]), device=device)
posterior_steps = int(parser["MCMC"]["G_SAMPLE_STEPS"])
posterior_s = torch.tensor(float(parser["MCMC"]["G_STEP_SIZE"]), device=device)


def update_step(x, grad_f, s):
    """Update the current state of the sampler."""
    return  x + (s * s * grad_f) + torch.sqrt(torch.tensor(2)) * s * torch.randn_like(x)


def sample_p0():
    """Sample from the prior distribution."""

    return p0_sig * torch.randn(*[batch_size, z_channels, 1, 1], device=device, requires_grad=True)


def sample_prior(EBM):
    """
    Sample from the prior distribution.

    Args:
    - EBM: energy-based model

    Returns:
    - z: latent space variable sampled from p_a(x)
    """

    z = sample_p0()

    for k in range(prior_steps):
        grad_f = prior_grad_log(z, EBM)
        z = update_step(z, grad_f, prior_s)

    return z


def sample_posterior(x, EBM, GEN, temp_schedule):
    """
    Sample from the posterior distribution.

    Args:
    - x: batch of data samples
    - EBM: energy-based model
    - GEN: generator
    - temp_schedule: temperature schedule

    Returns:
    - z_samples: samples from the posterior distribution indexed by temperature
    """

    z_samples = torch.empty(len(temp_schedule), batch_size, z_channels, 1, 1, device=device)

    for idx, t in enumerate(temp_schedule):
        z = sample_p0()

        for k in range(posterior_steps):
            grad_f = posterior_grad_log(z, x, t, EBM, GEN)
            z = update_step(z, grad_f, posterior_s)

        z_samples[idx] = z

    return z_samples
