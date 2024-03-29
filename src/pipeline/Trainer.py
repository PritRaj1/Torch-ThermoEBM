import torch
import lightning as L
import torch.nn as nn
import configparser
import numpy as np
from torchvision.utils import make_grid

from src.pipeline.loss_fcn import ThermodynamicIntegrationLoss, VanillaLoss
from src.networks.PriorModel import EBM
from src.networks.GeneratorModel import GEN
from src.MCMC_Sampling.sample_distributions import sample_prior
from src.pipeline.metrics import profile_image, stores_grads

torch.autograd.set_detect_anomaly(True)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

E_lr = float(parser["OPTIMIZER"]["E_LR"])
G_lr = float(parser["OPTIMIZER"]["G_LR"])
temp_power = int(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])


class LatentEBM_Model(L.LightningModule):
    def __init__(self, image_dim):
        super().__init__()

        self.automatic_optimization = False

        self.EBM = EBM()
        self.GEN = GEN(image_dim)

        if temp_power == 0:
            self.loss_fcn = VanillaLoss
        else:
            self.loss_fcn = ThermodynamicIntegrationLoss

    def configure_optimizers(self):
        # Optimisers
        EBM_optimiser = torch.optim.Adam(self.EBM.parameters(), lr=E_lr)
        GEN_optimiser = torch.optim.Adam(self.GEN.parameters(), lr=G_lr)

        return [EBM_optimiser, GEN_optimiser]

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Get loss
        loss_EBM, loss_GEN = self.loss_fcn(
            x, self.EBM, self.GEN
        )

        # Update params
        self.optimiser_step(loss_EBM, loss_GEN)

        #  Negative marginal log-likelihood
        loss = loss_GEN + loss_EBM

        self.log("train_loss", loss)
        stores_grads(self)

        return loss

    def validation_step(self, batch):

        # Make sure the model is not in eval mode, so that gradients are enabled
        with torch.enable_grad():
            x, _ = batch

            # Get loss
            loss_EBM, loss_GEN = self.loss_fcn(
                x, self.EBM, self.generate
            )
            loss = loss_GEN + loss_EBM

            self.log("val_loss", loss)
            stores_grads(self)

            # Generate data
            generated_data = self.generate()

            # Profile the image
            fid_score, mifid_score, kid_score, lpips_score = profile_image(
                x, generated_data
            )

            # Log the metrics
            self.log("val_loss", loss)
            self.log("val_FID", fid_score)
            self.log("val_MIFID", mifid_score)
            self.log("val_KID", kid_score)
            self.log("val_lpips", lpips_score)

            # Log a grid of 4x4 images
            tensorboard = self.logger.experiment
            grid = make_grid(generated_data[:16], nrow=4)
            tensorboard.add_image("Generated Images", grid, self.current_epoch)

        return loss

    def optimiser_step(self, lossE, lossG):
        """
        Method to perform the optimiser step for both the EBM and GEN networks.
        """
        EBM_opt, GEN_opt = self.optimizers()

        GEN_opt.zero_grad()
        self.manual_backward(lossG)
        GEN_opt.step()

        EBM_opt.zero_grad()
        self.manual_backward(lossE)
        EBM_opt.step()

    def generate(self, x=None):
        """
        Function to generate a batch of samples from the lightning model.
        """

        # Sample latent variable and exponentially tilt
        z_prior = sample_prior(self.EBM)

        with torch.no_grad():
            x_pred = self.GEN(z_prior)

        return x_pred
