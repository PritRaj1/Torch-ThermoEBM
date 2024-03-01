import torch
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser['SIGMAS']['p0_SIGMA'])
pl_sig = float(parser['SIGMAS']['LKHOOD_SIGMA'])


def prior_grad_log(z, EBM):
    """
    Function to compute the gradient of the log prior: log[p_a(x)] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - EBM: energy-based model

    Returns:
    - ∇_z( log[p_a(x)] )
    """

    f_z = EBM(z)

    grad_f = torch.autograd.grad(f_z.sum(), z, create_graph=True)[0]

    return grad_f - (z / (p0_sig**2))

def posterior_grad_log(
    z, x, t, EBM, GEN
):
    """
    Function to compute the gradient of the log posterior: log[ p(x | z)^t * p(z) ] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - x: batch of data samples
    - t: current temperature
    - EBM: energy-based model 
    - GEN: generator 

    Returns:
    - ∇_z( log[p_θ(z | x)] ) ∝ ∇_z( log[p(x | z)^t * p(z)] )
    """


    g_z = GEN(z.view(z.size(0), -1, 1, 1))
    log_llhood = - t *(torch.norm(x-g_z, dim=-1)**2) / (2.0 * pl_sig **2 )
    grad_log_llhood = torch.autograd.grad(log_llhood.sum(), z, create_graph=True)[0]

    grad_prior = prior_grad_log(z, EBM)

    return grad_log_llhood + grad_prior
    
