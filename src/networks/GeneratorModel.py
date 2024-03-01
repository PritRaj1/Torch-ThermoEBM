import configparser
import torch.nn as nn

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

feature_dim = int(parser['GEN']['GEN_FEATURE_DIM'])
output_dim = int(parser['GEN']['CHANNELS'])
leak_coef = float(parser['GEN']['GEN_LEAK'])

class GEN(nn.Module):
    def __init__(self, input_dim, image_dim):
        super().__init__()


        f = nn.LeakyReLU(leak_coef)

        # Top-down generator network
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_dim * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_dim * 16),
            f,
            nn.ConvTranspose2d(feature_dim * 16, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            f,
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            f,
        )

        if image_dim == 64: # 64 x 64 images
            self.layers.add_module('layer_4', nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1))
            self.layers.add_module('layer_5', f)
            self.layers.add_module('layer_6', nn.ConvTranspose2d(feature_dim * 2, output_dim, kernel_size=4, stride=2, padding=1))
            self.layers.add_module('layer_7', nn.Tanh())
        else: # 32 x 32 images
            self.layers.add_module('layer_4', nn.ConvTranspose2d(feature_dim * 4, output_dim, kernel_size=4, stride=1, padding=1))
            self.layers.add_module('layer_5', nn.Tanh())
    
    def forward(self, z):
        return self.layers(z)