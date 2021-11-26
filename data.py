from numpy import DataSource
import pandas as pd
from torch.utils import data
from torch.utils.data import dataloader
import matplotlib.pyplot as plt

def getdata():
    train_start = "2000/1/3"
    train_end = "2010/12/31"
    test_start = "2011/1/3"
    test_end = "2021/11/11"
    data = pd.read_csv("data.csv", index_col="Date")
    data = data.sort_values("Date")
    train_data = data.loc[train_start:train_end] 
    test_data = data.loc[test_start:test_end]
    return train_data, test_data

def data_plot():
    data = data = pd.read_csv("data.csv", index_col="Date")
    data = data[["TPX", "SPX"]]
    data = data.sort_values("Date").loc["2001/1/3":"2021/11/11"]
    data = data.values
    print(type(data))
    print(data)
    #data.plot()
    #plt.show()
    
data_plot()

