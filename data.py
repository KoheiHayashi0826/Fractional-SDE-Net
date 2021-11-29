from matplotlib import colors
import numpy as np
from numpy.core.fromnumeric import transpose
import numpy.random as npr
from numpy import DataSource
import pandas as pd
from torch.utils import data
from torch.utils.data import dataloader
import matplotlib.pyplot as plt



def generate_spiral2d(nspiral=2,
                      ntotal=50,  # sample number to plot smooth graph
                      nsample=10, # ( ntotal - 2*nsample ) must be positive
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=False):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    #zs_cw = stop + 1. - orig_ts
    #rs_cw = a + b * 50. / zs_cw
    #xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    #orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        #plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        #cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc #if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    #print(type(samp_trajs))
    #print(samp_trajs)
    #from data import getdata
    #data1, data2 = getdata()
    #samp_trajs = data1.values
    #print(orig_trajs.shape)
    print(samp_trajs.shape)
    #print(orig_trajs)
    print(samp_trajs)
    #plt.plot(samp_ts, samp_trajs[1])
    plt.scatter(samp_trajs[0], samp_trajs[1])
    plt.show()

    return samp_trajs, orig_ts, samp_ts

#generate_spiral2d()








def get_TOPIX_data(batch_dim=10,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=10):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """
    train_start = "2019/1/3"
    train_end = "2020/12/31"
    test_start = train_end #"2021/1/3"
    test_end = "2021/11/11"
    data = pd.read_csv("data.csv", index_col="Date")
    data = data.sort_values("Date")
    data = data["TPX"]
    orig_data = data.loc[train_start:test_end]
    train_data = data.loc[train_start:train_end].values 
    test_data = data.loc[test_start:test_end].values
    ntotal = orig_data.size + 1
    ntrain = train_data.size
    ntest = test_data.size

    orig_ts = np.linspace(start, stop, num=ntotal)
    train_ts = orig_ts[:ntrain]
    test_ts = orig_ts[ntrain:ntrain+ntest]

    sample_traj = train_data

    #if []:
    #    plt.figure()
    #    plt.plot(orig_traj[:, 0], orig_traj[:, 1], label='counter clock')
    #   plt.legend()
    #    plt.savefig('./ground_truth.png', dpi=500)
    #    print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    sample_trajs = []
    for _ in range(batch_dim):
        # don't sample t0 very near the start or the end
        #t0_idx = npr.multinomial(
        #    1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        #t0_idx = np.argmax(t0_idx) + nsample
        #orig_trajs.append(orig_traj)

        #samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        sample_traj = train_data.copy()
        sample_traj += npr.randn(*sample_traj.shape) * noise_std
        sample_trajs.append(sample_traj)
    #print(samp_trajs)
    #print(samp_traj)
    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    #orig_trajs = np.stack(orig_trajs, axis=0)
    sample_trajs = np.stack(sample_trajs, axis=0)
    """
    plt.scatter(train_ts, train_data, label="train", s=3)
    plt.scatter(test_ts, test_data, label="test", s=3)
    plt.legend()
    plt.show()
    """
    return sample_trajs, train_data, test_data, train_ts, test_ts


def data_plot():
    data = data = pd.read_csv("data.csv", index_col="Date")
    data = data[["TPX", "SPX", "SX5E"]]
    data = data.sort_values("Date").loc["1987/1/3":"2021/11/11"]
    #data = data.values
    #print(type(data))
    #print(data)
    data.plot()
    plt.show()
    
#data_plot()
