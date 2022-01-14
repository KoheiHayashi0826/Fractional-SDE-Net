"""Read csv data of generated paths and evaluate ACF score. 
 Export summary as csv file"""
from itertools import product
import os 

import math
from posixpath import split
import numpy as np
#from numpy import linalg
from numpy.core.arrayprint import printoptions
import pandas as pd
from pandas.core.base import PandasObject
pd.options.display.float_format = '{:.3f}'.format
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

from train import DICT_DATANAME, DICT_DATANAME_STOCK, DICT_DATANAME_fOU, DICT_METHOD 
from train import stock_ts_points, fOU_ts_points
from utils.plots import plot_generated_paths, plot_hist, plot_correlogram, plot_scatter

DICT_EVALUATION = ['Hurst', 'Distribution', 'ACF_annealed', 'wACF_annealed', 'R2 Score', 'ACF_quenched', 'wACF_quenched']
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

    if data_name in DICT_DATANAME_STOCK:
        paths_gen = stock_transform(data_csv.values[:,2:], eval_obj)
        paths_gen_train = stock_transform(data_csv.loc[train_start:train_end].values[:,2:], eval_obj)
        paths_gen_test = stock_transform(data_csv.loc[test_start:test_end].values[:,2:], eval_obj) 
        path_hist = stock_transform(data_csv[[data_name]].values, eval_obj).reshape(-1)
        path_hist_train = stock_transform(data_csv[[data_name]].loc[train_start:train_end].values, eval_obj).reshape(-1)
        path_hist_test = stock_transform(data_csv[[data_name]].loc[test_start:test_end].values, eval_obj).reshape(-1)
    elif data_name in DICT_DATANAME_fOU:
        split_pt = int(ts_points[1])
        paths_gen = stock_transform(data_csv.values[:,2:], eval_obj)
        paths_gen_train = stock_transform(data_csv.values[:split_pt,2:], eval_obj)
        paths_gen_test = stock_transform(data_csv.values[split_pt:,2:], eval_obj) 
 
        path_hist = stock_transform(data_csv[[data_name]].values, eval_obj).reshape(-1)
        path_hist_train = stock_transform(data_csv[[data_name]].values[:split_pt:], eval_obj).reshape(-1)
        path_hist_test = stock_transform(data_csv[[data_name]].values[split_pt:], eval_obj).reshape(-1)


    return paths_gen, paths_gen_train, paths_gen_test, path_hist, path_hist_train, path_hist_test


def stock_transform(data: np.ndarray, type: str) -> np.ndarray:
    if type == 'price': pass
    elif type == 'return': data = np.diff(data, axis=0) 
    elif type == 'RV': data = np.square(np.diff(data, axis=0))
    return data

    
def acf_score(path_hist, paths_gen, weight):
    path_hist = np.abs(path_hist)
    paths_gen = np.abs(paths_gen)
    lag_horizon = path_hist.size - 500
    scores = []
    acf_hist = sm.tsa.stattools.acf(path_hist, nlags=lag_horizon, fft=False)
    for i in range(paths_gen.shape[1]):
        acf_gen = sm.tsa.stattools.acf(paths_gen[:,i], nlags=lag_horizon, fft=False)
        score = np.square(acf_hist - acf_gen).sum()**0.5 # l^2-norm 
        scores.append(score)
        
    return (np.mean(scores), np.std(scores))


def acf_score_annealed(path_hist, paths_gen, weight):
    batch_size = paths_gen.shape[1]
    path_hist = np.abs(path_hist)
    paths_gen = np.abs(paths_gen)
    lag_horizon = path_hist.size - 500

    if weight:
        w = np.arange(1, lag_horizon + 2)
        norm = np.square(w).astype(np.int64).sum()
        w = w / w.mean()
    else: 
        w = np.ones(lag_horizon + 1)
        #norm = np.square(w).sum()
        #w = w / norm

    acf_hist = sm.tsa.stattools.acf(path_hist, nlags=lag_horizon, fft=False) * w
    acfs_gen = []
    for i in range(paths_gen.shape[1]):
        acf_gen = sm.tsa.stattools.acf(paths_gen[:,i], nlags=lag_horizon, fft=False) * w
        acfs_gen.append(acf_gen)
    acfs_gen = np.stack(acfs_gen, axis=0)
    acf_gen_annealed = np.mean(acfs_gen.reshape(-1, batch_size), axis=1)
    score = np.square(acf_hist - acf_gen_annealed).sum()**0.5 # l^2-norm 
    return f'&{score:.3f}' 

def marginal_distribution_score(path_hist, paths_gen):
    """ Use Scott's choice to determine width: 3.5\sigma^2 n^(-1/3) = 0.01 """
    width = 0.01
    scores = []
    for i in range(paths_gen.shape[1]):
        min = np.min([np.min(path_hist), np.min(paths_gen[:,i])])
        max = np.max([np.max(path_hist), np.max(paths_gen[:,i])])
        bins = np.arange(np.floor(min/width).astype(int) , np.ceil(max/width).astype(int)+1) * width # np.arange([1,3]) = [1,2]
        emp_hist, bins = np.histogram(path_hist, bins, density=True)
        emp_gen, bins = np.histogram(paths_gen[:,i], bins, density=True)
        pdf_diff = np.abs(emp_hist - emp_gen) 
        score = np.sum(pdf_diff) * width
        scores.append(score)
    return (np.mean(scores), np.std(scores)) 


def prediction_score(path_hist, paths_gen):
    scores = []
    for i in range(paths_gen.shape[1]):
        score = r2_score(path_hist, paths_gen[:,i])
        scores.append(score)
    return (np.mean(scores), np.std(scores)) 


def estimate_hurst(data, name, method):

    if data.shape[0] == data.size: # when batch_size=1
        mean = np.mean(data)
        cumsum = np.cumsum(data - mean)
        sample_range = np.maximum.accumulate(cumsum) - np.minimum.accumulate(cumsum)
        std = np.cumsum(np.square(data -mean)) / np.arange(1, len(data) + 1)
        Q = sample_range / std 
        T = len(data) - 1  # max is (len-1)
        y = np.log(Q[-T:-1]).reshape(-1, 1)
        x = np.log(np.arange(len(data)- T + 2, len(data) + 1)).reshape(-1, 1)
        plot_scatter(name, method, x, y)
        reg = LinearRegression().fit(x, y)
        hurst = reg.coef_[0, 0]
        out = f'& {hurst:.3f}'
    else:
        scores = []
        for i in range(data.shape[1]):
            mean = np.mean(data[:,i])
            cumsum = np.cumsum(data[:,i] - mean)
            sample_range = np.maximum.accumulate(cumsum) - np.minimum.accumulate(cumsum)
            std = np.cumsum(np.square(data[:,i] - mean)) / np.arange(1, len(data[:,i]) + 1)
            Q = sample_range / std 
            T = len(data) - 1  # max is (len-1)
            y = np.log(Q[-T:-1]).reshape(-1, 1)
            x = np.log(np.arange(len(data[:,i])- T + 2, len(data[:,i]) + 1)).reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            hurst = reg.coef_[0, 0]
            scores.append(hurst)
        out = (np.mean(scores), np.std(scores)) 
        plot_scatter(name, method, x, y)
    return out

def print_error(list):
    out = f'& {list[0]:.3f} $\pm$ {list[1]:.3f}' 
    return out


def save_summary():
    summary = np.full((len(DICT_METHOD) + 1, len(DICT_EVALUATION)), "& -", dtype=object)
  
    for key_data in DICT_DATANAME:
        dir_name = "./result/" + key_data
        file_name = dir_name + "/summary.csv"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, key_method in enumerate(DICT_METHOD):
            if key_data in DICT_DATANAME_STOCK:
                paths_gen, paths_gen_train, paths_gen_test, path_hist, path_hist_train, path_hist_test = read_data(stock_ts_points, key_data, key_method)
            elif key_data in DICT_DATANAME_fOU:
                paths_gen, paths_gen_train, paths_gen_test, path_hist, path_hist_train, path_hist_test = read_data(fOU_ts_points, key_data, key_method)
            plot_correlogram(key_data, key_method, path_hist, paths_gen[:,0])
            summary[i, 0] = print_error(estimate_hurst(paths_gen, key_data, key_method))
            summary[i, 1] = print_error(marginal_distribution_score(path_hist, paths_gen))
            summary[i, 2] = acf_score_annealed(path_hist, paths_gen, weight=False)
            summary[i, 3] = acf_score_annealed(path_hist, paths_gen, weight=True)
            summary[i, 4] = print_error(prediction_score(path_hist_test, paths_gen_test))
            summary[i, 5] = print_error(acf_score(path_hist, paths_gen, weight=False))            
            summary[i, 6] = print_error(acf_score(path_hist, paths_gen, weight=True))
        summary[i+1, 0] = estimate_hurst(path_hist, key_data, 'original')
        summary_pd = pd.DataFrame(data=summary, columns=DICT_EVALUATION, index=DICT_METHOD + ['Original'])
        summary_pd.to_csv(file_name)
        print(f"Data: {key_data}")
        print(summary_pd)


if __name__=="__main__":
    save_summary()
