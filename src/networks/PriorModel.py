import torch.nn as nn
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

hidden_units = int(parser['EBM']['EBM_FEATURE_DIM'])
output_dim = int(parser['EBM']['Z_CHANNELS'])
leak_coef = float(parser['EBM']['EBM_LEAK'])

class EBM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()  

        f = nn.LeakyReLU(leak_coef)

        self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_units), 
                    f,
                    nn.Linear(hidden_units, hidden_units),
                    f,
                    nn.Linear(hidden_units, output_dim),
                )
        
    def forward(self, z):
        return self.layers(z.squeeze()).view(-1, self.output_dim, 1, 1).requires_grad_()