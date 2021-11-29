import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib

from data import get_TOPIX_data
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=50) # originally 5000
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



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


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


if __name__ == '__main__':
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 1

    batch_dim = 50
    start = 0.
    stop = 1.
    noise_std = 0.1
    
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate TOPIX data
    from data import get_TOPIX_data
    sample_trajs, train_data, test_data, train_ts, test_ts = get_TOPIX_data(
        batch_dim=batch_dim,
        start=start,
        stop=stop,
        noise_std=noise_std
    )
    #orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    sample_trajs = torch.from_numpy(sample_trajs).float().to(device)
    train_ts = torch.from_numpy(train_ts).float().to(device)
    test_ts = torch.from_numpy(test_ts).float().to(device)

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch_dim).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #orig_trajs = checkpoint['orig_trajs']
            sample_trajs = checkpoint['sample_trajs']
            #orig_ts = checkpoint['orig_ts']
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

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, train_ts).permute(1, 0, 2)
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
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'orig_trajs': orig_trajs,
                'sample_trajs': sample_trajs,
                #'orig_ts': orig_ts,
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
            #orig_ts = torch.from_numpy(orig_ts).float().to(device)

            # take first trajectory for visualization
            z0 = z0[0]

            #ts_pos = np.linspace(0., 2. * np.pi, num=2000)
            #ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
            #ts_pos = torch.from_numpy(ts_pos).float().to(device)
            #ts_neg = torch.from_numpy(ts_neg).float().to(device)

            zs_learn = odeint(func, z0, train_ts)
            zs_pred = odeint(func, zs_learn[-1], test_ts)
            xs_learn = dec(zs_learn)
            xs_pred = dec(zs_pred)
            

        xs_learn = xs_learn.cpu().numpy()
        xs_pred = xs_pred.cpu().numpy()
        #xs_neg = xs_neg.cpu().numpy()
        #orig_traj = orig_trajs[0].cpu().numpy()
        #samp_traj = samp_trajs[0].cpu().numpy()
        #samp_traj = sample_trajs[0].cpu().numpy()


        plt.figure()
        #plt.plot(orig_traj[:, 0], orig_traj[:, 1],
        #         'g', label='true trajectory')
        plt.plot(train_ts, xs_learn, #'r',
                 label='learned trajectory')
        plt.plot(test_ts, xs_pred, #'r',
                 label='predicted trajectory', ls="--")
        #plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
        #         label='learned trajectory (t<0)')
        
        plt.scatter(train_ts, train_data, label='train data', s=3)
        plt.scatter(test_ts, test_data, label='test data', s=3)
        plt.legend()
        plt.savefig('./vis.png', dpi=500)
        print('Saved visualization figure at {}'.format('./vis.png'))
    

