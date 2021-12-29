"""Read csv data of generated paths and evaluate ACF score. 
 Export summary as csv file"""
import os 

import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

#from data.data import data_plot
from train import DICT_DATANAME, DICT_METHOD


def acf_plot(data_name, method):
    
    dir_name = "../result/" + data_name + "/correlogram"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    head_end = -1 #2000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    path = f"../result/{data_name}/path_csv/{method}.csv"
    data_csv = pd.read_csv(path)
    data = data_csv[["Value"]].diff().dropna().head(head_end)

    data = data.values.reshape(-1)
    #data = np.abs(data)
    data_ori = data_csv[[data_name]].diff().dropna().head(head_end)
    data_ori = data_ori.values.reshape(-1)
    data_ori = np.abs(data_ori)
    
    plt.figure()
    autocorrelation_plot(pd.Series(data), label="Generated")
    autocorrelation_plot(pd.Series(data_ori), label="Historical")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.savefig(file_name, dpi=500)
    

if __name__=="__main__":
    
    for key_data in DICT_DATANAME:
        for key_method in DICT_METHOD:
            acf_plot(key_data, key_method)
