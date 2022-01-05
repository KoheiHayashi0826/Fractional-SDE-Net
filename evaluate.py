"""Read csv data of generated paths and evaluate ACF score. 
 Export summary as csv file"""
import os 

import numpy as np
#from numpy import linalg
from numpy.core.arrayprint import printoptions
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import statsmodels.api as sm

from train import DICT_DATANAME, DICT_METHOD
DICT_EVALUATION = ['ACF', 'MMD', 'R2']

def read_data(data_name, method):
    head_end = -1 #2000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    path = f"../result/{data_name}/path_csv/{method}.csv"
    data_csv = pd.read_csv(path)
    data_gen = data_csv[["Value"]].diff().dropna() #.head(head_end)
    data_gen = data_gen.values.reshape(-1)
    #data = np.abs(data)
    data_hist = data_csv[[data_name]].diff().dropna() #.head(head_end)
    data_hist = data_hist.values.reshape(-1)
    #data_hist = np.abs(data_hist)
    return data_hist, data_gen


def acf_plot(data_name, method, data_hist, data_gen):
    
    dir_name = "../result/" + data_name + "/correlogram"
    file_name = dir_name + f"/{method}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.figure()
    autocorrelation_plot(pd.Series(data_gen), label="Generated")
    autocorrelation_plot(pd.Series(data_hist), label="Historical", alpha=0.7) # alpha=0: totally transparent
    #plt.xscale("log")
    #plt.yscale("log")
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.savefig(file_name, dpi=500)

    
def acf_score(data_hist, data_gen):
    lag_horizon = 250
    acf_hist = sm.tsa.stattools.acf(data_hist, nlags=lag_horizon, fft=False)
    acf_gen = sm.tsa.stattools.acf(data_gen, nlags=lag_horizon, fft=False)
    score = np.square(acf_hist - acf_gen).sum()**0.5 # l^2-norm 
    return score    


def save_summary():
    summary = np.full((len(DICT_METHOD), len(DICT_EVALUATION)), "?", dtype=object)

    for key_data in DICT_DATANAME:
        dir_name = "../result/" + key_data
        file_name = dir_name + "/summary.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, key_method in enumerate(DICT_METHOD):
            data_hist, data_gen = read_data(key_data, key_method)
            acf_plot(key_data, key_method, data_hist, data_gen)
            summary[i, 0] = acf_score(data_hist, data_gen)
            #summary[i, 1] = acf_score(data_hist, data_gen)
            #summary[i, 2] = acf_score(data_hist, data_gen)
    summary_pd = pd.DataFrame(data=summary, columns=DICT_EVALUATION, index=DICT_METHOD)
    summary_pd.to_csv(file_name)
    print(summary_pd)

if __name__=="__main__":
    save_summary()
