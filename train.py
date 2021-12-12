import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
from tqdm import tqdm


matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.neural_net import LatentFSDEfunc, LatentODEfunc, RecognitionRNN, Decoder
from utils.neural_net import LatentSDEfunc, latent_dim, batch_dim
from utils.utils import log_normal_pdf, normal_kl, RunningAverageMeter
from utils.plots import plot_path, plot_hist
from utils.utils import save_csv
from data.data import get_stock_data

parser = argparse.ArgumentParser()
parser.add_argument('--ode_adjoint', type=eval, default=False)
parser.add_argument('--sde_adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=6) # originally 5000
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hurst', type=float, default=0.6)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

DICT_DATANAME = ["TPX"] #, "SPX", "SX5E"]
DICT_METHOD = ["fSDE"] #["ODE", "SDE", "fSDE"]


if args.ode_adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
if args.sde_adjoint:
    from torchsde import sdeint_adjoint as sdeint
else:
    from torchsde import sdeint
from utils.fsde_solver import fsdeint


def train(data_name, method):
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 1

    noise_std = 0.0001
    
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate data
    sample_trajs, train_data, test_data, train_ts_pd, test_ts_pd, train_ts, test_ts = get_stock_data(
        data_name=data_name, batch_dim=batch_dim)
    sample_trajs = torch.from_numpy(sample_trajs).float().to(device)
    train_ts = torch.from_numpy(train_ts).float().to(device)
    test_ts = torch.from_numpy(test_ts).float().to(device)

    # model
    # Call instance
    func_ODE = LatentODEfunc().to(device) #latent_dim, nhidden).to(device)
    func_SDE = LatentSDEfunc().to(device)
    func_fSDE = LatentFSDEfunc().to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_dim).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()
    
    for itr in tqdm(range(1, args.niters + 1)):
        optimizer.zero_grad()
        # backward in time to infer q(z_0)
        h = rec.initHidden().to(device)
        for t in range(sample_trajs.size(1)):
            obs = sample_trajs[:, t].reshape(-1, 1)
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean # dimension (batch_size, latent_size)

        # forward in time and solve differential equation for reconstructions
        if method=="ODE":
            pred_z = odeint(func_ODE, z0, train_ts).permute(1, 0, 2)
            pred_x = dec(pred_z).reshape(batch_dim, -1)
        elif method=="SDE":
            pred_z = sdeint(func_SDE, z0, train_ts).permute(1, 0, 2)
            pred_x = dec(pred_z).reshape(batch_dim, -1)
        elif method=="fSDE":
            # dimension of fsdeint is (batch_size, t_size, latent_size)
            pred_z = fsdeint(func_fSDE, hurst=args.hurst, y0=z0, ts=train_ts) #.permute(0, 2, 1)
            pred_x = dec(pred_z).reshape(batch_dim, -1)

        # compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(
            sample_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        #if itr%5==0:
        #    print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))
    print(f'Training complete after {itr} iters.\n')


    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        h = rec.initHidden().to(device)
        for t in range(sample_trajs.size(1)):
            obs = sample_trajs[:, t].reshape(-1, 1)
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        if method=="ODE":
            z0 = z0[0]
            zs_learn = odeint(func_ODE, z0, train_ts)
            zs_pred = odeint(func_ODE, zs_learn[-1,:], test_ts)
            xs_learn = dec(zs_learn)
            xs_pred = dec(zs_pred)
        elif method=="SDE":
            zs_learn = sdeint(func_SDE, z0, train_ts)
            zs_pred = sdeint(func_SDE, zs_learn[-1,:,:], test_ts)
            xs_learn = dec(zs_learn[:,0,:])
            xs_pred = dec(zs_pred[:,0,:])
        elif method=="fSDE":
            zs_learn = fsdeint(func_fSDE, hurst=args.hurst, y0=z0, ts=train_ts)
            zs_pred = fsdeint(func_fSDE, hurst=args.hurst, y0=zs_learn[:,-1,:], ts=test_ts) 
            xs_learn = dec(zs_learn[0,:,:])
            xs_pred = dec(zs_pred[0,:,:])
            

    xs_learn = xs_learn.cpu().numpy()
    xs_pred = xs_pred.cpu().numpy()
    save_csv(data_name, method, train_ts_pd, xs_learn.reshape(-1))
    plot_path(data_name, method, train_ts, xs_learn, test_ts, xs_pred, train_data, test_data)
    plot_hist(data_name, method, xs_learn, train_data)



if __name__ == '__main__':
    for key_data in DICT_DATANAME:
        for key_method in DICT_METHOD:
            print(f"Training begin with data:{key_data}, method:{key_method}") 
            train(data_name = key_data, method=key_method)

