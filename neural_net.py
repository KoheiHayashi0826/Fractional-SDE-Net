import numpy as np
from numpy.core.fromnumeric import put
import torch
import torch.nn as nn

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden) # fully connected
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
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


batch_size, state_size, brownian_size = 10, 4, 1

class LatentSDEfunc(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, nhidden=20):
        super().__init__()
        self.drift_fc1 = nn.Linear(state_size, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_size)
        self.drift_elu = nn.ELU()
        
        self.diff_fc1 = nn.Linear(state_size, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_size * brownian_size)
        self.diff_elu = nn.ELU()

    # Drift
    def f(self, t, y):
        out = self.drift_fc1(y)
        out = self.drift_elu(out)
        out = self.drift_fc2(out)
        out = self.drift_elu(out)
        out = self.drift_fc3(out)
        out = self.drift_elu(out)
        return out #self.mu_linear(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        out = self.diff_fc1(y)
        out = self.diff_elu(out)
        out = self.diff_fc2(out)
        out = self.diff_elu(out)
        out = self.diff_fc3(out).view(batch_size, state_size, brownian_size)
        return out


class LatentFSDEfunc(nn.Module):
    #def __init__(self):
    #    pass

    def drift(t, y):
        nhidden = 20
        y = torch.from_numpy(y).float()
        drift_fc1 = nn.Linear(state_size, nhidden)
        drift_fc2 = nn.Linear(nhidden, nhidden)
        drift_fc3 = nn.Linear(nhidden, state_size)
        drift_elu = nn.ELU()
        out = drift_fc1(y)
        out = drift_elu(out)
        out = drift_fc2(out)
        out = drift_elu(out)
        out = drift_fc3(out)
        out = out.detach().numpy()
        return out  

    def diffusion(t, y):
        nhidden= 20
        y = torch.from_numpy(y).float()
        diff_fc1 = nn.Linear(state_size, nhidden)
        diff_fc2 = nn.Linear(nhidden, nhidden)
        diff_fc3 = nn.Linear(nhidden, state_size)
        diff_elu = nn.Tanh()
        out = diff_fc1(y)
        out = diff_elu(out)
        out = diff_fc2(out)
        out = diff_elu(out)
        out = diff_fc3(out)
        out = out.detach().numpy()
        return out

#print(type(LatentFSDEfunc.drift))

#input = torch.ones(state_size)
#list = range(4)
#input = np.array(list)
#print(type(input))
#output = LatentFSDEfunc.diffusion(t=0, y=input)
#print(output)

