#from typing import Tuple
import itertools

from fbm import FBM
import numpy as np
from pandas.io.stata import stata_epoch
#from numpy.core.fromnumeric import put
import torch
#from torch._C import T
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Linear


bool_xavier_normal = True
init_gain_sde = 1 #2
init_gain_fsde = 1.5 #2.5
batch_dim, state_dim, bm_dim = 100, 1, 1
nhidden_rnn, nhidden_sde, nhidden_fsde = 40, 20, 20
latent_dim = 1


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20, gain=1):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden) # fully connected
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        if bool_xavier_normal:
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


class GeneratorRNN(nn.Module):
    """
      h: hidden-layer variable
      x: observed variable
    """
    def __init__(self, obs_dim=state_dim, nhidden=nhidden_rnn):
        super(GeneratorRNN, self).__init__()
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, obs_dim)
        self.h2h_out = nn.Linear(nhidden, obs_dim)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        h_out = self.h2h_out(h)
        x_out = self.h2o(h)
        return x_out, h_out # both has state_dim

    #def initHidden(self):
    #    return torch.zeros(self.nbatch, self.nhidden)


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
#batch_dim, latent_dim, bm_dim = 3, 2, 1

class LatentSDEfunc(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, nhidden=nhidden_sde, state_dim=state_dim, gain=init_gain_sde):
        super().__init__()
        #self.nhidden = nhidden
        #self.latent_dim = latent_dim
        #self.bm_dim = bm_dim
        #self.batch_dim = batch_dim

        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)
        
        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim) # * bm_dim)
        
        self.act = nn.Tanh() #(inplace=True)

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    # Drift
    def f(self, t, y):
        out = self.drift_fc1(y)
        out = self.act(out)
        out = self.drift_fc2(out)
        out = self.act(out)
        out = self.drift_fc3(out)
        #out = self.act(out)
        return out  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        out = self.diff_fc1(y)
        out = self.act(out)
        out = self.diff_fc2(out)
        out = self.act(out)
        out = self.diff_fc3(out)
        #out = self.act(out)
        return out.view(batch_dim, state_dim, bm_dim) #.view(self.batch_dim, self.latent_dim, self.bm_dim)


class LatentFSDEfunc(nn.Module):

    def __init__(self, nhidden=nhidden_fsde, state_dim=state_dim, gain=init_gain_fsde):
        super(LatentFSDEfunc, self).__init__()
        self.drift_fc1 = nn.Linear(state_dim, nhidden)
        self.drift_fc2 = nn.Linear(nhidden, nhidden)
        self.drift_fc3 = nn.Linear(nhidden, state_dim)
        #self.drift_act = nn.Tanh() #(inplace=True)
        #self.act = nn.Tanh() #(inplace=True)
        
        self.diff_fc1 = nn.Linear(state_dim, nhidden)
        self.diff_fc2 = nn.Linear(nhidden, nhidden)
        self.diff_fc3 = nn.Linear(nhidden, state_dim)
        #self.diff_act = nn.Tanh() #(inplace=True)
        
        self.act = nn.Tanh() #(inplace=True)

        if bool_xavier_normal:
            nn.init.xavier_normal_(self.drift_fc1.weight, gain)
            nn.init.xavier_normal_(self.drift_fc2.weight, gain)
            nn.init.xavier_normal_(self.drift_fc3.weight, gain)
            nn.init.xavier_normal_(self.diff_fc1.weight, gain)
            nn.init.xavier_normal_(self.diff_fc2.weight, gain)
            nn.init.xavier_normal_(self.diff_fc3.weight, gain)
        
    def drift(self, y):
        out = self.drift_fc1(y)
        out = self.act(out)
        out = self.drift_fc2(out)
        out = self.act(out)
        out = self.drift_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  

    def diffusion(self, y):
        out = self.diff_fc1(y)
        out = self.act(out)
        out = self.diff_fc2(out)
        out = self.act(out)
        out = self.diff_fc3(out)
        return out #.reshape(batch_dim, latent_dim)  




"""
# Following will not be used. 
class FSDENet(nn.Module):

    def __init__(self, nhidden=2, state_size=latent_dim, gain=init_gain):
        super(FSDENet, self).__init__()
        self.drift_fc1 = nn.Linear(state_size, nhidden)
        self.drift = nn.Sequential(

            nn.Linear(state_size, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, state_size),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(state_size, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, state_size)
        )
        if boole_xavier_normal:
            for param in FSDENet.parameters():
                nn.init.xavier_normal_(param, gain)
                #nn.init.xavier_normal_(self.diffusion.weight, gain)
         
    def forward(self, hurst, x0, ts):
        batch_size = x0.size(0)
        state_size = x0.size(1)
        t_start = float(ts[0])
        t_end = float(ts[-1])
        nsteps = 5000
        dt = (t_end - t_start) / nsteps
        
        #y = torch.tile(x0, (nsteps + 1,)).reshape(batch_size, nsteps + 1, state_size).clone()
        #y = torch.zeros(batch_size, nsteps + 1, state_size)
        #print(y.shape)
        y = []
        #dB = []
        #for i, k in itertools.product(range(batch_size), range(nsteps+1)): 
        for i in range(batch_size):
            y.append(x0[i])
            B = FBM(n=nsteps, hurst=hurst, length=t_end-t_start).fbm()
            dB = np.diff(B) if i==0 else np.append(dB, np.diff(B)) 
            for k in range(1, nsteps+1):    
                index = k + i * (nsteps + 1)
                y_next = y[index-1] + self.drift(y[index-1]) * dt \
                    + self.diffusion(y[index-1]) * dB[index-1-(k-1)]
                y.append(y_next)
                #y[i,k] = y[i,k-1] + dB[k-1] + self.drift(y[i,k-1]) * dt \
                    #+ self.diffusion(y[i,k-1]) * dB[k-1] # "dB[k]=B[k+1]-B[k]"    
        #print(torch.stack(y))
        y = torch.stack(y, axis=0).reshape(batch_size, -1, state_size) 
        print(y)
 
        ts_panel = torch.tile(ts, (batch_size, state_size)).reshape(batch_size, state_size, -1).permute(0, 2, 1) 
        num = (ts_panel - t_start) / (t_end -t_start) * nsteps 
        n_floor = torch.floor(num).to(torch.int64) 
        n_ceil = torch.ceil(num).to(torch.int64)
        ts_floor = t_start + n_floor * dt
        solution = y[:,n_floor[0,:,0]] + (ts_panel - ts_floor) * (y[:,n_ceil[0,:,0]] - y[:,n_floor[0,:,0]]) / dt 
        #print(solution[:,:3])
        return solution
"""
