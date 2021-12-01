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
from utils import log_normal_pdf, normal_kl


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=100) # originally 5000
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    from torchsde import sdeint_adjoint as sdeint
else:
    from torchdiffeq import odeint
    from torchsde import sdeint



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



if __name__ == '__main__':
    latent_dim = state_size
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 1

    batch_dim = batch_size
    start = 0.
    stop = 1.
    noise_std = 0.1
    
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate TOPIX data
    from data import get_TOPIX_data
    sample_trajs, train_data, test_data, train_ts, test_ts = get_TOPIX_data(
        batch_dim=batch_dim)
#        start=start,
#        stop=stop,
#        noise_std=noise_std
#    )
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
            #func_SDE.load_state_dict(checkpoint['func_state_dict'])
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
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve sde for reconstructions
            # sdeint_out has shepe (t_size, batch_size, state_size)
            pred_z = sdeint(func_SDE, z0, train_ts).permute(1, 0, 2)
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

            zs_learn = sdeint(func_SDE, z0, train_ts)
            #print(zs_learn.size())
            zs_pred = sdeint(func_SDE, zs_learn[-1,:,:], test_ts)
            xs_learn = dec(zs_learn[:,0,:])
            xs_pred = dec(zs_pred[:,0,:])
            
        xs_learn = xs_learn.cpu().numpy()
        xs_pred = xs_pred.cpu().numpy()

        plt.figure()
        plt.plot(train_ts, xs_learn, #'r',
                 label='learned trajectory')
        plt.plot(test_ts, xs_pred, #'r',
                 label='predicted trajectory', ls="--")
        
        plt.scatter(train_ts, train_data, label='train data', s=3)
        plt.scatter(test_ts, test_data, label='test data', s=3)
        plt.legend()
        plt.savefig('./vis_SDE.png', dpi=500)
        print('Saved visualization figure at {}'.format('./vis_SDE.png'))
    


"""
#batch_size, state_size, brownian_size = 32, 3, 2
t_size = 100



sde = LatentSDEfunc()
y0 = torch.full((batch_size, state_size), 0.1)
ts = torch.linspace(0, 1, t_size)
# Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
# ys will have shape (t_size, batch_size, state_size)
ys = sdeint(sde, y0, ts).permute(0, 2, 1)
print(ys.size())
"""
