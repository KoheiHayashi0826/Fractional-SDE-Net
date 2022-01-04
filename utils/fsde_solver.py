#  Implementation of fSDESolver for fBm (H>1/2)
#  We use the Euler scheme whose error is o(n^{-2H+1}) (see [Nourdin, 2007])  
from os import stat, times
from numpy.core.function_base import linspace
from numpy.random import set_state
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

#from utils.neural_net import LatentFSDEfunc

#device = "cpu" 
#torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"


"""
class SolveFractionalSDEMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, x0, ts):
        ctx.save_for_backward(x0)
        h = x0 / 4 
        y = 4 * h 
        return y
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x, = ctx.saved_tensors
        h = x / 4 
        dy_dx = d_tanh(h)
        grad_inputs = grad_outputs * dy_dx
        return grad_inputs

def d_tanh(x):
    return 1 / (x.cosh() ** 2)
"""




def fsdeint(func_fSDE, hurst, y0, ts):
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
    state_size = y0.size(1)
    t_start = float(ts[0])
    t_end = float(ts[-1])
    nsteps = 5000
    dt = (t_end - t_start) / nsteps

    for i in range(batch_size):
        B = FBM(n=nsteps, hurst=hurst, length=t_end-t_start).fbm()
        dB = np.diff(B) if i ==0 else np.append(dB, np.diff(B))
    dB = dB.reshape(batch_size, nsteps)
    dB = torch.from_numpy(dB)
    dB = torch.tile(dB, (state_size,)).reshape(batch_size, state_size, -1).permute(0, 2, 1)
    #print(dB.shape)


    y = torch.tile(y0, (nsteps + 1,)).reshape(batch_size, nsteps + 1, state_size) #.requires_grad_() #.clone()
    #print(y)
    #print(ts, dB)
    #lin = nn.Linear(batch_size*state_size, batch_size*state_size)
    #print(y[:,0].shape)
    #print(y[0,0].reshape(batch_size*state_size))
    y_copy = torch.zeros(batch_size, nsteps + 1, state_size)
    y_copy[:,0] = y[:,0].clone()
    for k in range(1, nsteps + 1): # range(n) = [0, ..., n-1]
        y[:,k,:] = y[:,k-1,:] + func_fSDE.drift((k-1)*dt, y[:,k-1,:].reshape(batch_size*state_size)) * dt \
             + func_fSDE.diffusion((k-1)*dt, y[:,k-1,:].reshape(batch_size*state_size)) * dB[:,k-1] # "dB[k]=B[k+1]-B[k]"     
        y_copy[:,k] = y[:,k].clone()
    ts_panel = torch.tile(ts, (batch_size, state_size)).reshape(batch_size, state_size, -1).permute(0, 2, 1) 
    
    num = (ts_panel - t_start) / (t_end -t_start) * nsteps 
    n_floor = torch.floor(num).to(torch.int64) 
    n_ceil = torch.ceil(num).to(torch.int64)
    ts_floor = t_start + n_floor * dt
    solution = torch.zeros(state_size, ts.size(0), 1) #.requires_grad_()
    solution = y_copy[:,n_floor[0,:,0]] + (ts_panel - ts_floor) * (y_copy[:,n_ceil[0,:,0]] - y_copy[:,n_floor[0,:,0]]) / dt 
    #print(solution.shape)
    
    """
    with torch.no_grad():
        ts = ts.to(device).detach().numpy().copy()
        y0 = y0.to(device).detach().numpy().copy()
        
    y = np.tile(y0, nsteps + 1).reshape(batch_size, nsteps + 1, state_size)
    
    #for k in range(1, nsteps + 1): # range(n) = [0, ..., n-1]
    #    y[:,k] = y[:,k-1] + func_fSDE.drift((k-1)*dt, y[:,k-1]) * dt \
    #        + func_fSDE.diffusion((k-1)*dt, y[:,k-1]) * dB[k-1] # "dB[k]=B[k+1]-B[k]"     
    
    #ts = np.tile(ts, (batch_size , state_size)).reshape(batch_size, state_size, ts.size).transpose(0, 2, 1)

    num = (ts - t_start) / (t_end -t_start) * nsteps 
    n_floor = np.floor(num).astype(int)
    n_ceil = np.ceil(num).astype(int)
    ts_floor = t_start + n_floor * dt
    
    y = y[:,n_floor[0,:,0]] + np.multiply(ts - ts_floor, y[:,n_ceil[0,:,0]] - y[:,n_floor[0,:,0]]) / dt 
    y.tolist()
    y = torch.tensor(y, requires_grad=True).float()
    #print(solution)
    """
    return solution


"""
def experiment():
    func_fSDE = nn.Linear()
    list = range(7) # = [0, ..., 100]
    with torch.no_grad():
        ts = torch.tensor(list)
        y0 = torch.rand(5, 4)
    y = fsdeint(func_fSDE, hurst=0.5, y0=y0, ts=ts)
"""
