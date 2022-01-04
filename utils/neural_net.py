from typing import Tuple
import numpy as np
#from numpy.core.fromnumeric import put
import torch
#from torch._C import T
import torch.nn as nn
from torch.nn import init

boole_xavier_normal = True
init_gain = 2


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, gain=init_gain):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden) # fully connected
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        if boole_xavier_normal:
            nn.init.xavier_normal_(self.fc1.weight, gain)
            nn.init.xavier_normal_(self.fc2.weight, gain)
            nn.init.xavier_normal_(self.fc3.weight, gain)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    """
      h: hidden-layer variable
      x: observed variable
    """
    def __init__(self, latent_dim=4, obs_dim=1, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=1, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperperameters for SDE- and fSDE-Net
batch_dim, latent_dim, bm_dim = 20, 2, 1
#batch_dim, latent_dim, bm_dim = 1, 1, 1

class LatentSDEfunc(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, nhidden=20, latent_dim=latent_dim, bm_dim=bm_dim, batch_dim=batch_dim, gain=init_gain):
        super().__init__()
        self.nhidden = nhidden
        self.latent_dim = latent_dim
        self.bm_dim = bm_dim
        self.batch_dim = batch_dim

        self.drift_fc1 = nn.Linear(latent_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, latent_dim)
        self.drift_act = nn.ELU() #(inplace=True)
        
        self.diff_fc1 = nn.Linear(latent_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, latent_dim * bm_dim)
        self.diff_act = nn.Tanh() #(inplace=True)

        if boole_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    # Drift
    def f(self, t, y):
        out = self.drift_fc1(y)
        out = self.drift_act(out)
        #out = self.drift_fc2(out)
        #out = self.drift_elu(out)
        out = self.drift_fc3(out)
        out = self.drift_act(out)
        return out  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        out = self.diff_fc1(y)
        out = self.diff_act(out)
        #out = self.diff_fc2(out)
        #out = self.diff_elu(out)
        out = self.diff_fc3(out)
        out = self.diff_act(out)
        return out.view(self.batch_dim, self.latent_dim, self.bm_dim)


class LatentFSDEfunc(nn.Module):

    def __init__(self, nhidden=20, latent_dim=latent_dim, gain=init_gain):
        super(LatentFSDEfunc, self).__init__()
        self.drift_fc1 = nn.Linear(batch_dim*latent_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, batch_dim*latent_dim)
        self.drift_fc3 = nn.Linear(nhidden, batch_dim*latent_dim)
        self.drift_act = nn.Tanh() #(inplace=True)
        
        self.diff_fc1 = nn.Linear(batch_dim*latent_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, batch_dim*latent_dim)
        self.diff_fc3 = nn.Linear(nhidden, batch_dim*latent_dim)
        self.diff_act = nn.Tanh() #(inplace=True)

        if boole_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        

    def drift(self, t, y):
        out = self.drift_fc1(y)
        out = self.drift_act(out)
        out = self.drift_fc2(out)
        #out = self.drift_act(out)
        #out = self.drift_fc3(out)
        return out.reshape(batch_dim, latent_dim)  

    def diffusion(self, t, y):
        out = self.diff_fc1(y)
        out = self.diff_act(out)
        out = self.diff_fc2(out)
        #out = self.diff_act(out)
        #out = self.diff_fc3(out)
        return out.reshape(batch_dim, latent_dim)  

