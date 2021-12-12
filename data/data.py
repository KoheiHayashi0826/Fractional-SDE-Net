from posixpath import curdir
from matplotlib import colors
import numpy as np
#from numpy.core.fromnumeric import transpose
import numpy.random as npr
#from numpy import DataSource
import pandas as pd
#from torch.utils import data
#from torch.utils.data import dataloader
import matplotlib.pyplot as plt

import os


def get_stock_data(data_name="TPX", batch_dim=10,
                      start=0.,
                      stop=1, 
                      noise_std=10):

    train_start = "2010/1/4"
    train_end = "2015/12/31"
    test_start = train_end
    test_end = "2021/11/11"
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("data.csv", index_col="Date").iloc[::-1]
    data = data[[data_name]]
    #data = data[["SPX"]]
    #data = data[["SX5E"]]
    
    
    data = data.loc[train_start:test_end]
    #data = np.log(data)
    #data = data.diff()
    data = (data - data.values.mean()) / data.values.std()

    train_data = data.loc[train_start:train_end]
    test_data = data.loc[test_start:test_end]

    train_ts_pd = list(train_data.index)
    test_ts_pd = list(test_data.index)
    train_ts = pd.to_datetime(train_ts_pd)
    test_ts = pd.to_datetime(test_ts_pd)
    train_ts = train_ts.map(pd.Timestamp.timestamp).values
    test_ts = test_ts.map(pd.Timestamp.timestamp).values
    
    train_ratio = train_ts.size / (train_ts.size + test_ts.size)
    test_ratio = test_ts.size / (train_ts.size + test_ts.size)
    train_ts = start + (train_ts - train_ts[0]) / (train_ts[-1] - train_ts[0]) * train_ratio * (stop - start)
    test_ts = train_ts[-1] + (test_ts - test_ts[0]) / (test_ts[-1] - test_ts[0]) * test_ratio * (stop - start) 
 

    sample_traj = train_data.values

    # sample starting timestamps
    sample_trajs = []
    for _ in range(batch_dim):
        #samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        sample_traj = train_data.copy()
        sample_traj += npr.randn(*sample_traj.shape) * noise_std
        sample_trajs.append(sample_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    sample_trajs = np.stack(sample_trajs, axis=0).reshape(batch_dim, -1)

    return sample_trajs, train_data.values, test_data.values, train_ts_pd, test_ts_pd, train_ts, test_ts

sample_trajs, train_data, test_data, train_ts_pd, test_ts_pd, train_ts, test_ts = get_stock_data()
#print(train_ts.size, test_ts.size)
#print(len(train_ts_pd), len(test_ts_pd))


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
