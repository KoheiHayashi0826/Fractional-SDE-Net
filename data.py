from matplotlib import colors
import numpy as np
#from numpy.core.fromnumeric import transpose
import numpy.random as npr
#from numpy import DataSource
import pandas as pd
#from torch.utils import data
#from torch.utils.data import dataloader
import matplotlib.pyplot as plt




def get_TOPIX_data(batch_dim=10,
                      start=0.,
                      stop=100, 
                      noise_std=10):

    train_start = "2018/1/2"
    train_end = "2020/12/31"
    test_start = train_end
    test_end = "2021/11/11"
    data = pd.read_csv("data.csv", index_col="Date").iloc[::-1]
    data = data.loc[train_start:test_end]
    ts = list(data.index)
    ts = pd.to_datetime(ts)
    ts = ts.map(pd.Timestamp.timestamp).values
    ts_normal = start + (ts - ts[0]) / (ts[-1] - ts[0]) * (stop - start)
 

    data_key = data[["TPX"]]
    orig_data = data_key.loc[train_start:test_end].values
    train_data = data_key.loc[train_start:train_end].values 
    test_data = data_key.loc[test_start:test_end].values
    
    ntrain = train_data.size
    ntest = test_data.size

    train_ts = ts_normal[:ntrain]
    test_ts = ts_normal[ntrain-1:ntrain+ntest]

    sample_traj = train_data

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
    sample_trajs = np.stack(sample_trajs, axis=0).reshape(batch_dim, -1)
    """
    plt.scatter(train_ts, train_data, label="train", s=3)
    plt.scatter(test_ts, test_data, label="test", s=3)
    plt.legend()
    plt.show()
    """
    return sample_trajs, train_data, test_data, train_ts, test_ts

#sample_trajs, train_data, test_data, train_ts, test_ts = get_TOPIX_data()
#print(sample_trajs.shape, train_ts, test_ts)



def data_plot():
    data_pd = pd.read_csv("data.csv", index_col="Date")
    data_pd = data_pd[["TPX"]]
    data_pd = data_pd.loc["2021/11/11":"2018/1/2"]
    data_pd = data_pd.iloc[::-1]
    #data_pd.plot()
    #plt.show()
    ts = list(data_pd.index)
    train_start = "2018/1/2"
    train_end = "2020/12/31"
    
    data = pd.read_csv("data.csv").iloc[::-1] 
    #ts = data["Date"]
    #print(ts)
    ts = pd.to_datetime(ts)
    ts = ts.map(pd.Timestamp.timestamp).values
    ts_normal = (ts - ts[0]) / (ts[-1] - ts[0])
    
    data_pd = data.set_index(["Date"])
    #print(data_pd[["TPX"]].loc[train_start:train_end])
      
    #print(ts_normal)
    
data_plot()
