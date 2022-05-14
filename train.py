import os
import argparse
import logging
import time
import sys

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import numpy as np
from numpy.core.arrayprint import printoptions
import numpy.random as npr
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.data import get_stock_data, get_fOU_data, get_other_data
from utils.neural_net import LatentFSDEfunc, LatentODEfunc, GeneratorRNN
from utils.neural_net import LatentSDEfunc, latent_dim, batch_dim, nhidden_rnn
from utils.utils import RunningAverageMeter, log_normal_pdf, normal_kl, calculate_log_likelihood
from utils.plots import plot_generated_paths, plot_original_path, plot_hist
from utils.utils import save_csv, tensor_to_numpy

#sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--ode_adjoint', type=eval, default=False)
parser.add_argument('--sde_adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=10) # originally 5000
parser.add_argument('--lr', type=float, default=0.04)
parser.add_argument('--reg_lambda', type=float, default=0)
parser.add_argument('--hurst', type=float, default=0.7)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_paths', type=int, default=10)
args = parser.parse_args()

DICT_DATANAME_STOCK = ["SPX", "TPX", "SX5E"]
#DICT_DATANAME_fOU = ['fOU_H0.7', 'fOU_H0.8', 'fOU_H0.9']
DICT_DATANAME_fOU = ['fOU_H0.7']
#DICT_DATANAME_OTHER = ['NileMin', 'ethernet', 'videoVBR', 'NBSdiff', 'NhemiTemp']
DICT_DATANAME_OTHER = ['NileMin', 'ethernet', 'NBSdiff', 'NhemiTemp']
#DICT_DATANAME = ['NileMin']
DICT_DATANAME = ['ethernet']
#DICT_DATANAME = DICT_DATANAME_OTHER
#DICT_DATANAME = DICT_DATANAME_STOCK + DICT_DATANAME_fOU + DICT_DATANAME_OTHER

#DICT_METHOD = ['fSDE']
DICT_METHOD = ['RNN', 'SDE', 'fSDE']

#ts_points = ['2010/1/4', '2020/12/31', '2021/11/11'] # train_start, train_end=test_start, test_end 
#ts_points = ['1986/4/10', '2015/12/31', '2021/11/11'] 
#ts_points = ['2000/1/3', '2020/12/31', '2021/11/11'] 
stock_ts_points = ['2000/1/3', '2020/12/31', '2021/11/11'] 
split_rate = 0.8
#fOU_ts_points = ['0', '900', '1000']
#other_ts_points = ['0', '600', '663']

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
 
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate data
    if data_name in DICT_DATANAME_STOCK:
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_stock_data(stock_ts_points, data_name)
    elif data_name in DICT_DATANAME_fOU: 
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_fOU_data(data_name, split_rate)   
    elif data_name in DICT_DATANAME_OTHER:
        train_data, test_data, train_ts_str, test_ts_str, train_ts, test_ts = get_other_data(data_name, split_rate)   
    train_data = torch.from_numpy(train_data).float().to(device) 
    test_data = torch.from_numpy(test_data).float().to(device)
    train_ts = torch.from_numpy(train_ts).float().to(device)
    test_ts = torch.from_numpy(test_ts).float().to(device) 
    
    ts_total = torch.cat((train_ts.reshape(-1), test_ts[1:]))
    data_total = torch.cat((train_data.reshape(-1), test_data.reshape(-1)[1:]))
    ts_total_str = list(train_ts_str) + list(test_ts_str[1:])

    # model
    # Call instance
    #rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_dim).to(device)
    #dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    #if method == "ODE":
    #    func_ODE = LatentODEfunc().to(device)
    #    params = (list(func_ODE.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    #elif method == "SDE":
    #    func_SDE = LatentSDEfunc().to(device)
    #    params = (list(func_SDE.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    #elif method == "fSDE":
    #    func_fSDE = LatentFSDEfunc().to(device)
    #    params = (list(func_fSDE.parameters())) 
    
    
    #params = []
    if method == "RNN":
        rnn = GeneratorRNN().to(device)
        params = list(rnn.parameters()) 
    elif method == "SDE":
        func_SDE = LatentSDEfunc().to(device)
        params = list(func_SDE.parameters()) 
    elif method == "fSDE":
        func_fSDE = LatentFSDEfunc().to(device)
        #fsdenet = FSDENet().to(device)
        params = (list(func_fSDE.parameters())) 
        #params = (list(fsdenet.parameters()))

    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()
    
    for itr in range(1, args.niters + 1): #tqdm(range(1, args.niters + 1)):
        optimizer.zero_grad()
        #h = rec.initHidden().to(device)
        #for t in range(sample_trajs.size(1)):
        #    obs = sample_trajs[:, t].reshape(-1, 1)
        #    out, h = rec.forward(obs, h)
        #qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        #epsilon = torch.randn(qz0_mean.size()).to(device)
        #z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean # dimension (batch_size, latent_size)
        #print(train_data.shape)
        z0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]
        #print(z0.shape)
        
        if method == "RNN":
            h = torch.randn(train_data.size(0), batch_dim, nhidden_rnn)
            z = z0
            pred_return = torch.zeros(batch_dim, latent_dim)
            for k in range(train_data.size(0)-1):        
                z, h_out = rnn(z, h[k])
                pred_return = torch.cat((pred_return, h_out), dim=1)
            pred_return = torch.cumsum(pred_return.unsqueeze(-1), dim=1)
            pred_z = torch.zeros(batch_dim, train_data.size(0), latent_dim) + train_data[0, 0] - pred_return
        elif method == "SDE":
            # dimension of sdeint is (t_size, batch_size, latent_size)
            pred_z = sdeint(func_SDE, z0, train_ts).permute(1, 0, 2)
        elif method == "fSDE":
            # dimension of fsdeint is (batch_size, t_size, latent_size)
            pred_z = fsdeint(func_fSDE, args.hurst, z0, train_ts) #.permute(0, 2, 1)
        
        # compute loss
        #noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        #noise_logvar = 2. * torch.log(noise_std_).to(device)
        #logpx = log_normal_pdf(
        #    sample_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
        #pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        #analytic_kl = normal_kl(qz0_mean, qz0_logvar,
        #                        pz0_mean, pz0_logvar).sum(-1)
        #loss = torch.mean(-logpx + analytic_kl, dim=0)
        #loss_meter.update(loss.item())
        #if itr%5==0:
        #    print('Iter: {}, loss: {:.4f}'.format(itr, -loss_meter.avg))

        with torch.autograd.set_detect_anomaly(True):
            loss = - calculate_log_likelihood(pred_z[:,:,0], train_data[:,0])
        
            reg_lambda = args.reg_lambda
            reg = torch.tensor(0.) 
            for param in params:
                reg += torch.norm(param, 1)
            loss += reg_lambda * reg

            loss.backward()
            optimizer.step()
        
        #if itr%5==0:
        print("Iter: {}, Log Likelihood: {:.4f}, Regularization: {:.4f}".format(itr, -loss, reg))        
    print(f'Training complete after {itr} iters.\n')
    
    
    # Generation of sample paths
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        #h = rec.initHidden().to(device)
        #for t in range(sample_trajs.size(1)):
        #    obs = sample_trajs[:, t].reshape(-1, 1)
        #    out, h = rec.forward(obs, h)
        #qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        #epsilon = torch.randn(qz0_mean.size()).to(device)
        #z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        x0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]
        #xs_gen = []

        
        if method == 'RNN':
            h = torch.randn(data_total.size(0), batch_dim, nhidden_rnn)
            x = x0
            return_pred = torch.zeros(batch_dim, latent_dim)
            for k in range(data_total.size(0)-1):        
                x, h_out = rnn(x, h[k])
                return_pred = torch.cat((return_pred, h_out), dim=1)
            return_pred = torch.cumsum(return_pred.unsqueeze(-1), dim=1)
            xs_gen = torch.zeros(batch_dim, data_total.size(0), latent_dim) + train_data[0, 0] - return_pred
        elif method == 'SDE':
            xs_gen = sdeint(func_SDE, x0, ts_total).permute(1, 0, 2)
        elif method == 'fSDE':
            xs_gen = fsdeint(func_fSDE, args.hurst, x0, ts_total)
        
        plot_original_path(data_name, ts_total, data_total)
        plot_generated_paths(min([args.num_paths, batch_dim]), data_name, method, ts_total, data_total, xs_gen)
        xs_gen_np = tensor_to_numpy(xs_gen[:,:,0]) 
        save_csv(data_name, method, ts_total_str, data_total.reshape(-1), xs_gen_np)
        plot_hist(data_name, method, xs_gen_np[0], train_data)


if __name__ == '__main__':
    for key_data in DICT_DATANAME:
        for key_method in DICT_METHOD:
            print(f"Training begin with data:{key_data}, method:{key_method}") 
            train(data_name = key_data, method=key_method)
    

