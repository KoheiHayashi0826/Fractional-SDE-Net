#  Implementation of fSDESolver for fBm (H>1/2)
#  We use the Euler scheme whose error is o(n^{-2H+1}) (see [Nourdin, 2007])  
from os import times
from numpy.core.function_base import linspace
import torch 
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
from neural_net import LatentFSDEfunc

device = "cpu" 
#torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
#device = "cuda" if torch.cuda.is_available() else "cpu"


def fsdeint(hurst, y0, ts):
    """ Solve fSDE of type "dy_t = b(t, y_t)dt + sigma(t, y_t)dB_t".
        First solve discrete steps and then linearly interpolate to return valued in ts.
    input:  
     hurst = Hurst parameter > 1/2
     y0 = list with dimension (batch_size, state_size), tensor
     ts = list of time points, tensor
    output
     y = y[ts], tensor, dimension = (batch_size, t_size, state_size)
    """
    batch_size = y0.size(0)
    latent_size = y0.size(1)
    
    with torch.no_grad():
        ts = ts.to(device).detach().numpy().copy()
        y0 = y0.to(device).detach().numpy().copy()
    
    t_start = float(ts[0])
    t_end = float(ts[-1])
    nsteps = 10
    #y_samples = []
    
    B = FBM(n=nsteps, hurst=hurst, length=t_end-t_start).fbm()
    dB = np.diff(B)
    dt = (t_end - t_start)/nsteps
    #ts = ts * nsteps / t_end
    #ts = np.floor(ts).astype(int)
    
    #for i in range(0, batch_size):
    y = np.tile(y0, nsteps + 1).reshape(batch_size, nsteps + 1, -1)
    #print(y[0].shape, y[1])
    for s in range(1, nsteps + 1): # range(n) = [0, ..., n-1]
        y[:,s] = y[:,s-1] + LatentFSDEfunc.drift(s*dt, y[:,s-1]) * dt \
            + LatentFSDEfunc.diffusion(s*dt, y[:,s-1]) * dB[s-1] # "dB[i]=B[i+1]-B[i]"     
    #print(y.shape)
    #print(plt.ylim)

    ts = np.tile(ts, (batch_size , latent_size)).reshape(batch_size, latent_size, ts.size).transpose(0, 2, 1)
    n_floor = np.floor( (ts - t_start) / dt ).astype(int)
    n_ceil = np.ceil( (ts - t_start) / dt ).astype(int)
    ts_floor = t_start + n_floor * dt
    #ts_ceil = t_start + n_ceil * dt
    #print(ts_ceil - ts_floor)

    y = y[:,n_floor[0,:,0]] + np.multiply(ts - ts_floor, y[:,n_ceil[0,:,0]] - y[:,n_floor[0,:,0]]) / dt 
    #print(y_sample.shape)
    #y_samples.append(y_sample)
    #y_samples = np.stack(y_samples, axis=0)
    y.tolist()
    y = torch.tensor(y, requires_grad=True).float()
    print(y)
    return y
    

def experiment():
    list = range(7) # = [0, ..., 100]
    with torch.no_grad():
        ts = torch.tensor(list)
        y0 = torch.rand(5, 4)
    y = fsdeint(hurst=0.5, y0=y0, ts=ts)
    #print(y, y.size())
    #y= y.detach().numpy().copy()
    
    #plt.hist(y[0,:,0], bins=50, color='red')
    #plt.hist(y[0,:,1], bins=50, color='blue')
    #plt.show()

   
def fbm_path(hurst):
    f = FBM(n=1024, hurst=hurst, length=1).fbm()
    ts = np.linspace(0, 1, 1025)
    plt.plot(ts, f, linewidth=3, label="H={}".format(hurst))
    #plt.xlabel("Time", fontsize=14)
    #plt.ylabel("Value", fontsize=14) 
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

#fbm_path(0.5)

experiment()


