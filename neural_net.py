import numpy as np
from numpy.core.fromnumeric import put
import torch
import torch.nn as nn

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
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


batch_size, state_size, brownian_size = 51, 4, 2

class LatentSDEfunc(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(state_size, state_size)
        self.sigma = nn.Linear(state_size, state_size * brownian_size)

    # Drift
    def f(self, t, y):
        return self.mu(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.sigma(y).view(batch_size, state_size, brownian_size)



class LatentFSDEfunc(nn.Module):
    #def __init__(self):
    #    pass

    def drift(t, y):
        y = torch.from_numpy(y).float()
        func_1 = nn.Linear(state_size, state_size)
        func_2 = nn.Tanh()
        out = func_1(y)
        out = func_2(out)
        out = func_1(out)
        out = out.detach().numpy()
        return out  

    def diffusion(t, y):
        y = torch.from_numpy(y).float()
        func = nn.Linear(state_size, state_size)
        out = func(y)
        out = out.detach().numpy()
        return out

#print(type(LatentFSDEfunc.drift))

#input = torch.ones(state_size)
list = range(4)
input = np.array(list)
#print(type(input))
output = LatentFSDEfunc.diffusion(t=0, y=input)
#print(output)

