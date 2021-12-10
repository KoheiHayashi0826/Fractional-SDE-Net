import matplotlib.pyplot as plt
import numpy as np
import os

def plot_path(data_name, method, train_ts, xs_learn, test_ts, xs_pred, train_data, test_data):
    dir_name = "./result/" + data_name + "/path_fig"
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
    dir_name = "./result/" + data_name + "/histgram"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    plt.figure()
    plt.hist(np.diff(xs_learn.reshape(-1)), alpha=0.7, bins=100, label='Generated')
    plt.hist(np.diff(train_data.reshape(-1)), alpha=0.5, bins=100, label='Historical')  
    plt.legend()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


