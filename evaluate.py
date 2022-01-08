"""Read csv data of generated paths and evaluate ACF score. 
 Export summary as csv file"""
import os 

import math
import numpy as np
#from numpy import linalg
from numpy.core.arrayprint import printoptions
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm

from train import DICT_DATANAME, DICT_METHOD, train, ts_points

DICT_EVALUATION = ['Distribution', 'ACF', 'R2 Score']

def read_data(ts_points, data_name, method):
    train_start = ts_points[0]
    train_end = test_start = ts_points[1]
    test_end = ts_points[2]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    path = f"./result/{data_name}/path_csv/{method}.csv"
    data_csv = pd.read_csv(path, index_col="Date")
    data_gen = data_csv[["0"]].diff().dropna() #.head(head_end)
    data_gen = data_gen.values.reshape(-1)
    train_data_gen = data_csv[["0"]].loc[train_start:train_end].diff().dropna() #.head(head_end)
    train_data_gen = train_data_gen.values.reshape(-1)
    test_data_gen = data_csv[["0"]].loc[test_start:test_end].diff().dropna() #.head(head_end)
    test_data_gen = test_data_gen.values.reshape(-1)
    #print(train_data_gen, test_data_gen)
    #data_gen = data_csv[["0"]].diff().dropna() #.head(head_end)
    #data_gen = data_gen.values.reshape(-1)
    #data = np.abs(data)
    data_hist = data_csv[[data_name]].diff().dropna() #.head(head_end)
    data_hist = data_hist.values.reshape(-1)
    train_data_hist = data_csv[[data_name]].loc[train_start:train_end].diff().dropna() #.head(head_end)
    train_data_hist = train_data_hist.values.reshape(-1)
    test_data_hist = data_csv[[data_name]].loc[test_start:test_end].diff().dropna() #.head(head_end)
    test_data_hist = test_data_hist.values.reshape(-1)
    #data_hist = np.abs(data_hist)
    return data_hist, data_gen, train_data_hist, train_data_gen, test_data_hist, test_data_gen


def acf_plot(data_name, method, data_hist, data_gen):
    
    dir_name = "./result/" + data_name + "/correlogram"
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


def marginal_distribution_score(data_hist, data_gen):
    """ Use Scott's choice to determine width: 3.5\sigma^2 n^(-1/3) = 0.01 """
    width = 0.01
    min = np.min([np.min(data_hist), np.min(data_gen)])
    max = np.max([np.max(data_hist), np.max(data_gen)])
    bins = np.arange(np.floor(min/width).astype(int) , np.ceil(max/width).astype(int)+1) * width # np.arange([1,3]) = [1,2]
    emp_hist, bins = np.histogram(data_hist, bins, density=True)
    emp_gen, bins = np.histogram(data_gen, bins, density=True)
    pdf_diff = np.abs(emp_hist - emp_gen) 
    score = np.sum(pdf_diff) * width
    return score    


def prediction_score(train_hist, train_gen, test_hist, test_gen):
    #print(test_gen.shape)
    #y_true, y_pred = test_hist, test_gen
    score = r2_score(test_hist, test_gen)
    #score = 0
    return score 


def save_summary():
    summary = np.full((len(DICT_METHOD), len(DICT_EVALUATION)), "?", dtype=object)

    for key_data in DICT_DATANAME:
        dir_name = "./result/" + key_data
        file_name = dir_name + "/summary.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, key_method in enumerate(DICT_METHOD):
            data_hist, data_gen, train_hist, train_gen, test_hist, test_gen = read_data(ts_points, key_data, key_method)
            acf_plot(key_data, key_method, data_hist, data_gen)
            summary[i, 0] = marginal_distribution_score(data_hist, data_gen)
            summary[i, 1] = acf_score(data_hist, data_gen)
            summary[i, 2] = prediction_score(train_hist, train_gen, test_hist, test_gen)
        summary_pd = pd.DataFrame(data=summary, columns=DICT_EVALUATION, index=DICT_METHOD)
        summary_pd.to_csv(file_name)
        print(f"Data: {key_data}")
        print(summary_pd)


if __name__=="__main__":
    save_summary()
