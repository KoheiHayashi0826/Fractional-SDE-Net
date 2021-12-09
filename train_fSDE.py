import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from neural_net import LatentODEfunc, RecognitionRNN, Decoder
from neural_net import LatentSDEfunc, state_size, batch_size
from utils import log_normal_pdf, normal_kl, RunningAverageMeter
from plots import plot_path, plot_hist
from utils import save_csv

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=6) # originally 5000
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hurst', type=float, default=0.6)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()


#if args.adjoint:
#    from torchdiffeq import odeint_adjoint as odeint
#    from torchsde import sdeint_adjoint as sdeint
#else:
#    from torchdiffeq import odeint
#    from torchsde import sdeint
from fsde_solver import fsdeint





if __name__ == '__main__':
    data_name = "TOPIX"
    latent_dim = state_size
    batch_dim = batch_size
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 1

    noise_std = 0.1
    
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate TOPIX data
    from data import get_TOPIX_data
    sample_trajs, train_data, test_data, train_ts_pd, test_ts_pd, train_ts, test_ts = get_TOPIX_data(
        batch_dim=batch_dim)
    sample_trajs = torch.from_numpy(sample_trajs).float().to(device)
    train_ts = torch.from_numpy(train_ts).float().to(device)
    test_ts = torch.from_numpy(test_ts).float().to(device)

    # model
    #func = LatentODEfunc(latent_dim, nhidden).to(device)
    func_SDE = LatentSDEfunc().to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_dim).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    #params = (list(func_SDE.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            sample_trajs = checkpoint['sample_trajs']
            train_ts = checkpoint['train_ts']
            test_ts = checkpoint['test_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in range(sample_trajs.size(1)):
                obs = sample_trajs[:, t].reshape(-1, 1)
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean # dimension is (batch_size, latent_size)
            #print(z0)

            # forward in time and solve ode for reconstructions
            # dimension of fsdeint is (batch_size, t_size, latent_size)
            pred_z = fsdeint(hurst=args.hurst, y0=z0, ts=train_ts) #.permute(0, 2, 1)
            #print(pred_z.size())
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
            if itr%5==0:
                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func_SDE.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'sample_trajs': sample_trajs,
                'train_ts': train_ts,
                'test_ts':test_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.visualize:
        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in range(sample_trajs.size(1)):
                obs = sample_trajs[:, t].reshape(-1, 1)
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # take first trajectory for visualization
            #z0 = z0[0]
            #print(z0.size())
            #z0 = torch.full((batch_size, state_size), z0[0])

            zs_learn = fsdeint(hurst=args.hurst, y0=z0, ts=train_ts)
            zs_pred = fsdeint(hurst=args.hurst, y0=zs_learn[:,-1,:], ts=test_ts-train_ts[-1])
            xs_learn = dec(zs_learn[0,:,:])
            xs_pred = dec(zs_pred[0,:,:])
            
        xs_learn = xs_learn.cpu().numpy()
        xs_pred = xs_pred.cpu().numpy()
        save_csv(data_name, "fSDE", train_ts_pd, xs_learn.reshape(-1))
        plot_path(data_name, "fSDE", train_ts, xs_learn, test_ts, xs_pred, train_data, test_data)
        plot_hist(data_name, "fSDE", xs_learn, train_data)
        


