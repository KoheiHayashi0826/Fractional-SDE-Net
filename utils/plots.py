import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import scipy.stats

def plot_generated_paths(num_paths, data_name, method, train_ts, train_data, xs_gen):
    dir_name = "../result/" + data_name + "/path_fig"
    file_name = dir_name + f"/{data_name}_path_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    plt.grid()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    plt.plot(train_ts, train_data) #, ls="--") #, label='train data')
    for i in range(num_paths):
        plt.plot(train_ts, xs_gen[i,:,0]) #, label='learned trajectory')
    #plt.plot(train_ts, xs_gen[-1])
    plt.xlim(0, 1)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Log path', fontsize=14)
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()
    #print('Saved visualization figure at {}'.format(file_name))


def plot_original_path(data_name, ts, data):
    dir_name = "../result/" + data_name + "/path_fig"
    file_name = dir_name + f"/{data_name}_path_original.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    plt.grid()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    plt.plot(ts, data) #, ls="--", label='train data')
    plt.xlim(ts[0], ts[-1])
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Log path', fontsize=14)
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()
    #print('Saved visualization figure at {}'.format(file_name))


def plot_hist(data_name, method, xs_learn, train_data):
    log_plot=False

    dir_name = "../result/" + data_name + "/histogram"
    file_name = dir_name + f"/{data_name}_histogram_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    data = xs_learn.reshape(-1)
    data = np.diff(data) 
    s = scipy.stats.skew(data)
    k = scipy.stats.kurtosis(data)
    
    data_ori = train_data.reshape(-1)
    data_ori = np.diff(data_ori)
    s_ori = scipy.stats.skew(data_ori)
    k_ori = scipy.stats.kurtosis(data_ori)

    nbins = 200

    if log_plot:
        fig = plt.figure(figsize=(16, 8))
        fig1 = fig.add_subplot(1, 2, 1)
        fig2 = fig.add_subplot(1, 2, 2)

        fig1.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True) 
        fig1.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True)
        fig1.set_title("pdf", fontsize=16) #(f"pdf, skew={s:.02f}, kurtosis={k:.02f}") 
        fig1.set_aspect('auto', 'box')

        fig2.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True, log=True)  
        fig2.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True, log=True)
        fig2.set_title("log-pdf", fontsize=16) #(f"log-pdf, skew={s_ori:.02f}, kurtosis={k_ori:.02f}") 
    
        fig1.legend(fontsize=14)
        fig2.legend(fontsize=14)

        plt.figure()
        plt.grid()
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.tight_layout()
        plt.savefig(file_name, dpi=500)
        plt.close()
    else:
        plt.figure()
        plt.grid()
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)  
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')

        plt.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True) 
        plt.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True)
        plt.xlim(-0.12, 0.12)
        plt.legend(fontsize=14)
        plt.title(f'{method}', fontsize=14)
        plt.xlabel('Log-return', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.tight_layout()
        plt.savefig(file_name, dpi=500)
        plt.close()


def plot_correlogram(data_name, method, data_hist, data_gen):
    
    dir_name = "./result/" + data_name + "/correlogram"
    file_name = dir_name + f"/{data_name}_correlogram_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data_hist, data_gen = abs(data_hist), abs(data_gen)
    
    plt.figure()
    plt.grid()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    autocorrelation_plot(pd.Series(data_hist), label="Historical", alpha=0.7) # alpha=0: totally transparent
    autocorrelation_plot(pd.Series(data_gen), label="Generated", alpha=0.7)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Autocorrelation', fontsize=14)
    plt.legend(fontsize=14)
    plt.title(f'{method}', fontsize=14)
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()


           
def plot_scatter(data_name, method, x, y):
    
    dir_name = "./result/" + data_name + "/RS_statistics"
    file_name = dir_name + f"/{data_name}_RS_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    plt.figure()
    plt.grid()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)  
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

    plt.scatter(x, y)
    plt.title(f'{method}', fontsize=14)
    plt.xlabel('logT', fontsize=14)
    plt.ylabel('log(R/S)', fontsize=14)
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    plt.close()
