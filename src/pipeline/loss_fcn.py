import configparser
import torch

from src.MCMC_Sampling.sample_distributions import sample_prior, sample_posterior
from torch.nn.functional import mse_loss

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])

if temp_power > 0:
    print("Using Temperature Schedule with Power: {}".format(temp_power))
    temp_schedule = torch.linspace(0, 1, num_temps) ** temp_power
    print("Temperature Schedule: {}".format(temp_schedule))
else:
    print("Using no Thermodynamic Integration, defaulting to Vanilla Model")



def ebm_loss(z_prior, z_posterior, EBM):
    """
    Function to compute energy-difference loss for the EBM model.
    """

    # Compute the energy of the posterior sample
    en_pos = EBM(z_posterior.detach()).mean()

    # Compute the energy of the prior sample
    en_neg = EBM(z_prior.detach()).mean()

    # Return the difference in energies
    return (en_pos - en_neg)


def gen_loss(x, z, GEN):
    """
    Function to compute MSE loss for the GEN model.
    """

    # Compute -log[ p_β(x | z) ]; max likelihood training
    x_pred = GEN(z) + (pl_sig * torch.randn_like(x))
    log_lkhood = mse_loss(x_pred, x) / (2.0 * pl_sig**2)

    return log_lkhood

def VanillaLoss(x, EBM, GEN):
    """
    Function to compute the energy-based model loss using Vanilla Monte Carlo.

    Args:
    - x: batch of data
    - EBM: energy-based model forward
    - GEN: generator forward pass

    Returns:
    - total_loss_gen: the total loss for the entire thermodynamic integration loop, log(p_β(x | z))
    """
    z_prior = sample_prior()
    z_posterior = sample_posterior(x, 1, EBM, GEN)
    loss_EBM = ebm_loss(z_prior, z_posterior, EBM)
    loss_GEN = gen_loss(x, z_posterior, GEN)

    return loss_EBM.sum(), loss_GEN.sum()

def ThermodynamicIntegrationLoss(x, EBM, GEN):
    """
    Function to compute the energy-based model loss using Thermodynamic Integration.

    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    Args:
    - x: batch of data
    - EBM: energy-based model forward
    - GEN: generator forward pass
    - temp_schedule: temperature schedule

    Returns:
    - 2x total_loss_gen: the total loss for the entire thermodynamic integration loop, log(p_β(x | z))
    """
    # Initialise at t=0
    total_loss = 0
    z_init = sample_posterior(x, 0, EBM, GEN)
    prev_loss = gen_loss(x, z_init, GEN)

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = torch.cat((torch.tensor([0]), temp_schedule))

    for i in range(1, len(temp_schedule)):

        # Find likelihood at current temperature
        z_posterior_t = sample_posterior(x, temp_schedule[i], EBM, GEN)
        current_loss = gen_loss(x, z_posterior_t, GEN)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (current_loss + prev_loss) * delta_T

        prev_loss = current_loss

    return total_loss.sum(), total_loss.sum()