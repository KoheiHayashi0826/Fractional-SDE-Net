#  Implementation of fSDESolver for fBm (H>1/2)
#  We use the Euler scheme whose error is o(n^{-2H+1}) (see [Nourdin, 2007])  
from numpy.core.function_base import linspace
import torch 
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
from neural_net import LatentFSDEfunc

#device = "cpu" 
#torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"


def fsdeint(hurst, y0, ts):
    """solve fSDE of type "dy_t = b(t, y_t)dt + sigma(t, y_t)dB_t" 
    input:  
     hurst = Hurst parameter > 1/2
     y0 = list with dimension (batch_size, state_size), tensor
     ts = list of time points, tensor
    output
     y = y[ts], tensor, dimension = (batch_size, t_size, state_size)
    """
    batch_size = y0.size(0)
    latent_size = y0.size(1)
    #print(batch_size)
    #y0 = y0[0]

    with torch.no_grad():
        ts = ts.to(device).detach().numpy().copy()
        y0 = y0.to(device).detach().numpy().copy()
    
    t0 = float(ts[0])
    t_end = float(ts[-1])
    nsteps = 500
    y_samples = []
    
    B = FBM(n=nsteps, hurst=hurst, length=t_end-t0).fbm()
    dB = np.diff(B)
    dt = (t_end - t0)/nsteps
    ts = ts * nsteps / t_end
    ts = np.floor(ts).astype(int)
        
    for i in range(0, batch_size):
        y = np.full((nsteps + 1, latent_size) ,y0[i])
        for s in range(1, nsteps + 1):
            y[s] = y[s-1] + LatentFSDEfunc.drift(s*dt, y[s-1])*dt \
                + LatentFSDEfunc.diffusion(s*dt, y[s-1]) * dB[s-1] # "dB[i]=B[i+1]-B[i]" 
        y_sample = y[ts]
        y_samples.append(y_sample)
    y_samples = np.stack(y_samples, axis=0)
    y_samples.tolist()
    y_samples = torch.tensor(y_samples, requires_grad=True)
    
    return y_samples

def experiment():
    list = [0, 1, 2, 3, 5, 6, 10]
    list = range(100)
    with torch.no_grad():
        ts = torch.tensor(list)
        y0 = torch.ones(3, 4)
    y = fsdeint(hurst=0.5, y0=y0, ts=ts)
    print(y, y.size())
    y= y.detach().numpy().copy()
    
    plt.hist(y[0,:,0], bins=50, color='red')
    plt.hist(y[0,:,1], bins=50, color='blue')
    plt.show()

   
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

#experiment()


