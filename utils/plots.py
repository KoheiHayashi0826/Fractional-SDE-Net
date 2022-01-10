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
    plt.plot(train_ts, train_data, ls="--") #, label='train data')
    for i in range(num_paths):
        plt.plot(train_ts, xs_gen[i,:,0]) #, label='learned trajectory')
    #plt.plot(train_ts, xs_gen[-1])
    #plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


def plot_original_path(data_name, ts, data):
    dir_name = "../result/" + data_name + "/path_fig"
    file_name = dir_name + f"/{data_name}_path_original.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    plt.plot(ts, data) #, ls="--", label='train data')
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    #print('Saved visualization figure at {}'.format(file_name))


def plot_hist(data_name, method, xs_learn, train_data):
    log_plot=False

    dir_name = "../result/" + data_name + "/histogram"
    file_name = dir_name + f"/{data_name}_histogram_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    

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

    nbins = 100

    if log_plot:
        fig = plt.figure(figsize=(16, 8))
        fig1 = fig.add_subplot(1, 2, 1)
        fig2 = fig.add_subplot(1, 2, 2)

        fig1.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True)
        fig1.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True) 
        fig1.set_title("pdf", fontsize=14) #(f"pdf, skew={s:.02f}, kurtosis={k:.02f}") 
        fig1.set_aspect('auto', 'box')

        fig2.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True, log=True)
        fig2.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True, log=True)  
        fig2.set_title("log-pdf", fontsize=14) #(f"log-pdf, skew={s_ori:.02f}, kurtosis={k_ori:.02f}") 
    
        fig1.legend(fontsize=20)
        fig2.legend(fontsize=20)
        plt.savefig(file_name, dpi=500)
        #print('Saved visualization figure at {}'.format(file_name))
    else:
        plt.figure()
        plt.hist(data, alpha=0.7, bins=nbins, label='Generated', density=True)
        plt.hist(data_ori, alpha=0.7, bins=nbins, label='Historical', density=True) 
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(file_name, dpi=500)
        
            
def plot_correlogram(data_name, method, data_hist, data_gen):
    
    dir_name = "./result/" + data_name + "/correlogram"
    file_name = dir_name + f"/{data_name}_correlogram_{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data_hist, data_gen = abs(data_hist), abs(data_gen)
    
    plt.figure()
    autocorrelation_plot(pd.Series(data_gen), label="Generated", alpha=0.7)
    autocorrelation_plot(pd.Series(data_hist), label="Historical", alpha=0.7) # alpha=0: totally transparent
    #plt.xlabel(fontsize=14)
    #plt.yscale("log")
    plt.ylim(-0.25, 0.25)
    plt.legend(fontsize=14)
    plt.savefig(file_name, dpi=500)


