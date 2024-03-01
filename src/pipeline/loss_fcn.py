import configparser
import torch

from src.MCMC_Sampling.sample_distributions import sample_prior, sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


def ebm_loss(z_prior, z_posterior, EBM):
    """
    Function to compute energy-difference loss for the EBM model.
    """

    # Compute the energy of the posterior sample
    en_pos = EBM(z_posterior.detach())

    # Compute the energy of the prior sample
    en_neg = EBM(z_prior.detach())

    # Return the difference in energies
    return (en_pos - en_neg).squeeze().sum(axis=-1)


def gen_loss(x, z, GEN):
    """
    Function to compute MSE loss for the GEN model.
    """

    # Compute -log[ p_β(x | z) ]; max likelihood training
    x_pred = GEN(z) + (pl_sig * torch.randn_like(x))
    log_lkhood = (torch.norm(x - x_pred, dim=(2,3)) ** 2) / (2.0 * pl_sig**2)

    return log_lkhood.sum(axis=-1)


def TI_EBM_loss_fcn(x, EBM, GEN, temp_schedule):
    """
    Function to compute the energy-based model loss using Thermodynamic Integration.

    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    Args:
    - x: sample of x
    - EBM: energy-based model forward
    - GEN: generator forward pass
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_a(z))
    """

    total_loss = 0

    # Generate z_posterior for all temperatures
    z_posterior = sample_posterior(x, EBM, GEN, temp_schedule)

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = torch.cat((torch.tensor([0]), temp_schedule))

    for i in range(1, len(temp_schedule)):
        z_prior = sample_prior(EBM)

        z_posterior_t = z_posterior[i - 1]

        loss_current = ebm_loss(z_prior, z_posterior_t, EBM)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss


def TI_GEN_loss_fcn(x, EBM, GEN, temp_schedule):
    """
    Function to compute the generator loss using Thermodynamic Integration.

    Args:
    - x: batch of x samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM: energy-based model
    - GEN: generator
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_β(x | z))
    """

    total_loss = 0

    # Generate z_posterior for all temperatures
    z_posterior = sample_posterior(x, EBM, GEN, temp_schedule)

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = torch.cat((torch.tensor([0]), temp_schedule))

    for i in range(1, len(temp_schedule)):

        z_posterior_t = z_posterior[i - 1]

        # MSE between g(z) and x, where z ~ p_θ(z|x, t)
        loss_current = gen_loss(x, z_posterior_t, GEN)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss
