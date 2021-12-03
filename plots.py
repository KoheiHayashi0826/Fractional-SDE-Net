import matplotlib.pyplot as plt
import numpy as np


def plot_path(train_ts, xs_learn, test_ts, xs_pred, train_data, test_data, path):
    plt.figure()
    plt.plot(train_ts, xs_learn, #'r',
             label='learned trajectory')
    plt.plot(test_ts, xs_pred, #'r',
             label='predicted trajectory', ls="--")
    plt.scatter(train_ts, train_data, label='train data', s=3)
    plt.scatter(test_ts, test_data, label='test data', s=3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    print('Saved visualization figure at {}'.format(path))


def plot_hist(xs_learn, train_data, path):
    plt.figure()
    plt.hist(np.diff(xs_learn.reshape(-1)), alpha=0.7, bins=100, label='learned data')
    plt.hist(np.diff(train_data.reshape(-1)), alpha=0.5, bins=100, label='real data')  
    plt.legend()
    plt.savefig(path, dpi=500)
    print('Saved visualization figure at {}'.format(path))


