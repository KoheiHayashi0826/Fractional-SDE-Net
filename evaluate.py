"""Read csv data of generated paths and evaluate ACF score. 
 Export summary as csv file"""
import os 

import math
import numpy as np
#from numpy import linalg
from numpy.core.arrayprint import printoptions
import pandas as pd
from pandas.core.base import PandasObject
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

from train import DICT_DATANAME, DICT_METHOD, ts_points
from utils.plots import plot_generated_paths, plot_hist, plot_correlogram, plot_scatter

DICT_EVALUATION = ['Distribution', 'ACF', 'R2 Score', 'Hurst']
#eval_obj = 'price' 
eval_obj = 'return'
#eval_obj = 'RV'

def read_data(ts_points, data_name, method):
    train_start = ts_points[0]
    train_end = test_start = ts_points[1]
    test_end = ts_points[2]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))    
    path = f"./result/{data_name}/path_csv/{method}.csv"
    data_csv = pd.read_csv(path, index_col="Date")
    data_gen = pd_transform(data_csv[["0"]], eval_obj) 
    train_data_gen = pd_transform(data_csv[["0"]].loc[train_start:train_end], eval_obj)
    test_data_gen = pd_transform(data_csv[["0"]].loc[test_start:test_end], eval_obj) 
    data_hist = pd_transform(data_csv[[data_name]], eval_obj)
    train_data_hist = pd_transform(data_csv[[data_name]].loc[train_start:train_end], eval_obj)
    test_data_hist = pd_transform(data_csv[[data_name]].loc[test_start:test_end], eval_obj)

    return data_hist, data_gen, train_data_hist, train_data_gen, test_data_hist, test_data_gen


def pd_transform(data: PandasObject, type: str) -> np.ndarray:
    data = data.values.reshape(-1)
    if type == 'price': pass
    elif type == 'return': data = np.diff(data) 
    elif type == 'RV': data = np.square(np.diff(data))
    return data

    
def acf_score(data_hist, data_gen):
    data_hist = np.abs(data_hist)
    data_gen = np.abs(data_gen)

    lag_horizon = 500
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


def estimate_hurst(data, name, method):
    mean = np.mean(data)
    cumsum = np.cumsum(data - mean)

    range = np.maximum.accumulate(cumsum) - np.minimum.accumulate(cumsum)
    std = np.cumsum(np.square(data -mean)) / np.arange(1, len(data) + 1)
    Q = range / std 
    T = 2500
    y = np.log(Q[-T:-1]).reshape(-1, 1)
    x = np.log(np.arange(len(data)- T + 2, len(data) + 1)).reshape(-1, 1)
    plot_scatter(name, method, x, y)
    reg = LinearRegression().fit(x, y)
    hurst = reg.coef_[0, 0]
    return hurst


def save_summary():
    summary = np.full((len(DICT_METHOD) + 1, len(DICT_EVALUATION)), "-", dtype=object)

    for key_data in DICT_DATANAME:
        dir_name = "./result/" + key_data
        file_name = dir_name + "/summary.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, key_method in enumerate(DICT_METHOD):
            data_hist, data_gen, train_hist, train_gen, test_hist, test_gen = read_data(ts_points, key_data, key_method)
            plot_correlogram(key_data, key_method, data_hist, data_gen)
            summary[i, 0] = marginal_distribution_score(data_hist, data_gen)
            summary[i, 1] = acf_score(data_hist, data_gen)
            summary[i, 2] = prediction_score(train_hist, train_gen, test_hist, test_gen)
            summary[i, 3] = estimate_hurst(train_gen, key_data, key_method)
        summary[3, 3] = estimate_hurst(train_hist, key_data, 'Original')
        summary_pd = pd.DataFrame(data=summary, columns=DICT_EVALUATION, index=DICT_METHOD + ['Original'])
        summary_pd.to_csv(file_name)
        print(f"Data: {key_data}")
        print(summary_pd)


if __name__=="__main__":
    save_summary()
