import numpy as np
import torch 
import pandas as pd
import os

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


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


def save_csv(data_name, method, ts, data_ori, data):
    dir_name = "../result/" + data_name + "/path_csv"
    file_name = dir_name + f"/{method}.csv"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    data_pd = pd.DataFrame({"Date": ts, data_name: data_ori, "Value": data})
    data_pd.to_csv(file_name)
    #print('Saved generated path at {}'.format(file_name))


#save_csv("TOPIX", "ODE", [1, 2, 3], [4, 5, 2])

