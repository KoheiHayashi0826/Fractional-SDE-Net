import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import os

from data import getdata

TOPIX = getdata()
print(TOPIX)
