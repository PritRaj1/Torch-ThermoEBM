import configparser
import torch

# Metrics
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

fid = FrechetInceptionDistance(feature=64, normalize=True)  # FID metric
mifid = MemorizationInformedFrechetInceptionDistance(
    feature=64, normalize=True
)  # MI-FID metric
kid = KernelInceptionDistance(
    feature=64, subset_size=batch_size, normalize=True
)  # KID metric
lpips = LearnedPerceptualImagePatchSimilarity(
    net_type="alex", normalize=True
)  # LPIPS metric


def profile_image(x, x_pred):

    # Convert for [-1, 1] to [0, 1], image probailities
    x_metric = ((x + 1) / 2).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
    gen_metric = (x_pred + 1) / 2

    # FID score
    fid.update(x_metric, real=True)
    fid.update(gen_metric, real=False)
    fid_score = fid.compute()

    # MI-FID score
    mifid.update(x_metric, real=True)
    mifid.update(gen_metric, real=False)
    mifid_score = mifid.compute()

    # KID score
    kid.update(x_metric, real=True)
    kid.update(gen_metric, real=False)
    kid_score = kid.compute()[0]

    # LPIPS score
    lpips_score = lpips(x_metric, gen_metric)

    return fid_score, mifid_score, kid_score, lpips_score


def loss_FLOPS(LitTrainer, x):
    """
    Method to compute the FLOPS required to calculate the loss.
    """

    # Profile FLOPS for loss calculation
    flops_profiler = FlopsProfiler(LitTrainer)

    flops_profiler.start_profile()

    loss_EBM, loss_GEN = LitTrainer.get_loss(x)

    flops = flops_profiler.get_total_flops()
    flops_profiler.end_profile()

    LitTrainer.log("loss_flops", flops)


def stores_grads(LitTrainer):
    """
    Method to store the gradients of the EBM and GEN losses.
    """

    for name, param in LitTrainer.EBM.named_parameters():
        if param.requires_grad:
            try:
                EBM_grad = param.grad
            except:
                EBM_grad = None

    for name, param in LitTrainer.GEN.named_parameters():
        if param.requires_grad:
            try:
                GEN_grad = param.grad
            except:
                GEN_grad = None

    # Variance of all grads
    all_grad = torch.cat([EBM_grad.flatten(), GEN_grad.flatten()])
    LitTrainer.log("grad_var", torch.var(all_grad))
