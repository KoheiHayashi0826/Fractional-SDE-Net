from distutils.util import split_quoted
import os

#from posixpath import curdir
from matplotlib import colors
import numpy as np
#from numpy.core.fromnumeric import transpose
import numpy.random as npr
#from numpy import DataSource
import pandas as pd
#from torch.utils import data
#from torch.utils.data import dataloader
import matplotlib.pyplot as plt



def get_stock_data(ts_poits, data_name): #, start=0., stop=1): 

    train_start = ts_poits[0] # "2010/1/4"
    train_end = test_start = ts_poits[1] # "2020/12/31"
    test_end = ts_poits[2] # "2021/11/11"
    
    start, stop = 0., 1.
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("data.csv", index_col="Date").iloc[::-1].dropna()
    data = data[[data_name]]
    
    data = data.loc[train_start:test_end]
    data = np.log(data)
    data_np = data.values
    data = (data - data_np.mean()) / data_np.std()

    train_data = data.loc[train_start:train_end]
    test_data = data.loc[test_start:test_end]

    train_ts_str = list(train_data.index)
    test_ts_str = list(test_data.index)

    train_ts = pd.to_datetime(train_ts_str)
    test_ts = pd.to_datetime(test_ts_str)
    train_ts = train_ts.map(pd.Timestamp.timestamp).values
    test_ts = test_ts.map(pd.Timestamp.timestamp).values
    
    train_ratio = train_ts.size / (train_ts.size + test_ts.size)
    test_ratio = test_ts.size / (train_ts.size + test_ts.size)
    train_ts = start + (train_ts - train_ts[0]) / (train_ts[-1] - train_ts[0]) * train_ratio * (stop - start)
    test_ts = train_ts[-1] + (test_ts - test_ts[0]) / (test_ts[-1] - test_ts[0]) * test_ratio * (stop - start) 
 

    #sample_traj = train_data.values
    # sample starting timestamps
    #sample_trajs = []
    #for _ in range(batch_dim):
    #    #samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
    #    sample_traj = train_data.copy()
    #    sample_traj += npr.randn(*sample_traj.shape) * 1
    #    sample_trajs.append(sample_traj)
    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    #sample_trajs = np.stack(sample_trajs, axis=0).reshape(batch_dim, -1)

    return train_data.values, test_data.values, train_ts_str, test_ts_str, train_ts, test_ts


def get_fOU_data(name, split_rate):  
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv("fOU.csv")
    data_num = data['t'].size
    split_pt = round(split_rate * data_num)
    
    train_ts = data['t'].values[:split_pt] / data_num
    test_ts = data['t'].values[split_pt:] / data_num
    train_ts_str = train_ts.astype(object) #str(train_ts)
    test_ts_str = test_ts.astype(object) #str(test_ts)

    if name == 'fOU_H0.7':
        data = data['H=0.7'].values
    elif name == 'fOU_H0.8':
        data = data['H=0.8'].values
    elif name == 'fOU_H0.9':
        data = data['H=0.9'].values
    data = (data - np.mean(data)) / np.std(data)
    train_data = data[:split_pt]
    test_data = data[split_pt:]

    return train_data.reshape(-1, 1), test_data.reshape(-1, 1), train_ts_str, test_ts_str, train_ts, test_ts


def get_other_data(name, split_rate):  
    #split_pt = int(ts_points[1])
    #terminal_pt = int(ts_points[2])

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if name == 'NileMin':
        data = pd.read_csv("NileMin.csv")
    elif name == 'ethernet':
        data = pd.read_csv("ethernetTraffic.csv")
    elif name == 'NBSdiff':
        data = pd.read_csv("NBSdiff1kg.csv")
    elif name == 'NhemiTemp':
        data = pd.read_csv("NhemiTemp.csv")
    elif name == 'videoVBR':
        data = pd.read_csv("videoVBR.csv")
    data_num = data['t'].size
    split_pt = round(split_rate * data_num)
    
    train_ts = data['t'].values[:split_pt] / data_num
    test_ts = data['t'].values[split_pt:] / data_num
    train_ts_str = train_ts.astype(object) #str(train_ts)
    test_ts_str = test_ts.astype(object) #str(test_ts)

    data = data['x'].values
    data = (data - np.mean(data)) / np.std(data)
    data = np.cumsum(data)
    data = (data - np.mean(data)) / np.std(data)
    train_data = data[:split_pt]
    test_data = data[split_pt:]

    return train_data.reshape(-1, 1), test_data.reshape(-1, 1), train_ts_str, test_ts_str, train_ts, test_ts
