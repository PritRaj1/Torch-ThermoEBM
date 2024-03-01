import torch
import configparser

from src.MCMC_Sampling.grad_log_probs import prior_grad_log, posterior_grad_log

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser['SIGMAS']['p0_SIGMA'])
batch_size = int(parser['PIPELINE']['BATCH_SIZE'])
z_channels = int(parser['EBM']['Z_CHANNELS'])

prior_steps = int(parser['MCMC']['E_SAMPLE_STEPS'])
prior_s = float(parser['MCMC']['E_STEP_SIZE'])
posterior_steps = int(parser['MCMC']['G_SAMPLE_STEPS'])
posterior_s = float(parser['MCMC']['G_STEP_SIZE'])

def update_step(x, grad_f, s):
    """Update the current state of the sampler."""
    x += s * grad_f

    x += torch.sqrt(2 * s) * torch.randn_like(x)

    return x


def sample_p0():
    """Sample from the prior distribution."""

    return p0_sig * torch.randn(*[batch_size, z_channels, 1, 1], requires_grad=True)

def sample_prior(EBM):
    """
    Sample from the prior distribution.

    Args:
    - key: PRNG key
    - EBM: energy-based model

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_a(x)
    """

    key, z = sample_p0(key)

    for k in range(prior_steps):
        grad_f = prior_grad_log(z, EBM)
        key, z = update_step(z, grad_f, prior_s)

    return key, z

def sample_posterior(
    key,
    x, 
    EBM,
    GEN,
    temp_schedule
):
    """
    Sample from the posterior distribution.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM: energy-based model 
    - GEN: generator 
    - temp_schedule: temperature schedule

    Returns:
    - key: PRNG key
    - z_samples: samples from the posterior distribution indexed by temperature
    """

    z_samples = torch.zeros((len(temp_schedule), 1, 1, 1, z_channels))

    for idx, t in enumerate(temp_schedule):
        key, z = sample_p0(key)

        for k in range(posterior_steps):
            grad_f = posterior_grad_log(z, x, t, EBM, GEN)
            z = update_step(z, grad_f, posterior_s)

        z_samples[idx] = z

    return key, z_samples
