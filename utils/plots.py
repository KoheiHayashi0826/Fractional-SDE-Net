import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def plot_generated_paths(num_paths, data_name, method, train_ts, train_data, xs_gen):
    dir_name = "../result/" + data_name + "/path_fig"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    #num_paths = xs_gen.shape[0]

    plt.figure()
    plt.plot(train_ts, train_data, ls="--") #, label='train data')
    for i in range(num_paths):
        plt.plot(train_ts, xs_gen[i,:,0]) #, label='learned trajectory')
    #plt.plot(train_ts, xs_gen[-1])
    #plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


def plot_path(data_name, method, train_ts, xs_learn, test_ts, xs_pred, train_data, test_data):
    dir_name = "../result/" + data_name + "/path_fig"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    plt.figure()
    plt.plot(train_ts, xs_learn, label='learned trajectory')
    plt.plot(test_ts, xs_pred, label='predicted trajectory') #, ls="--")
    #plt.scatter(train_ts, train_data, label='train data', s=3)
    #plt.scatter(test_ts, test_data, label='test data', s=3)
    plt.plot(train_ts, train_data, label='train data')
    plt.plot(test_ts, test_data, label='test data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


def plot_hist(data_name, method, xs_learn, train_data):
    dir_name = "../result/" + data_name + "/histgram"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    nbins = 100
    fig = plt.figure(figsize=(16, 8))
    fig1 = fig.add_subplot(1, 2, 1)
    fig2 = fig.add_subplot(1, 2, 2)

    data = xs_learn.reshape(-1)
    data = np.diff(data) # calculate return 
    #data = (data - np.mean(data)) / np.std(data)
    s = scipy.stats.skew(data)
    k = scipy.stats.kurtosis(data)
    
    data_ori = train_data.reshape(-1)
    data_ori = np.diff(data_ori)
    #data_ori = (data_ori - np.mean(data_ori)) / np.std(data_ori)
    s_ori = scipy.stats.skew(data_ori)
    k_ori = scipy.stats.kurtosis(data_ori)

    fig1.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True)
    fig1.hist(data_ori, alpha=0.5, bins=nbins, label='Historical', density=True) 
    fig1.set_title(f"pdf, skew={s:.02f}, kurtosis={k:.02f}") 
    #fig1.text(0, 0, "a")

    fig2.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True, log=True)
    fig2.hist(data_ori, alpha=0.5, bins=nbins, label='Historical', density=True, log=True)  
    fig2.set_title(f"log-pdf, skew={s_ori:.02f}, kurtosis={k_ori:.02f}") 
    
    fig1.legend()
    fig2.legend()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


