import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.lib.type_check import real
import torch 
import pandas as pd
import os
from torch.functional import Tensor
from torch.nn.functional import batch_norm


def tensor_to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.
    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()

def log_normal_pdf(x, mean, var):
    x_norm = (x - mean)**2 / var
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    logvar = torch.log(var)
    pdf = -.5 * (const + logvar + x_norm)
    return pdf


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def save_csv(data_name, method, ts_str, data_ori, data):
    dir_name = "../result/" + data_name + "/path_csv"
    file_name = dir_name + f"/{method}.csv"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    data_pd = pd.DataFrame({"Date": ts_str, data_name: data_ori})
    for i in range(data.shape[0]):
        data_pd[str(i)] = data[i]
    data_pd.to_csv(file_name)
    #print('Saved generated path at {}'.format(file_name))


def calculate_log_likelihood(sample_paths, real_path):
    #num_partition = 4
    #batch_size = sample_paths.size(0)
    #real_paths = torch.tile(real_path, (batch_size,)).reshape(batch_size, -1)
    #real_paths_lower = torch.floor(num_partition*real_paths) / num_partition
    #real_paths_upper = torch.ceil(num_partition*real_paths)/ num_partition
    #L_sign = torch.sign((sample_paths - real_paths_lower) * (sample_paths - real_paths_upper))
    #L = torch.sum((1 - L_sign) / 2, 0) / batch_size # Likelihood vector
    #return torch.sum(L) #.prod(L) #.log()

    mean = torch.mean(torch.diff(sample_paths, dim=1), dim=0) 
    var = torch.var(torch.diff(sample_paths, dim=1), dim=0)
    x = torch.diff(real_path)
    log_pdf = log_normal_pdf(x, mean, var)
    return log_pdf.mean()
    
