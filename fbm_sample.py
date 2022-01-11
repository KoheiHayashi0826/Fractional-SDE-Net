import os

import matplotlib.pyplot as plt
import numpy as np
from fbm import FBM

def plot_fBm(hurst):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dir_name = "./result/fBm_sample"
    file_name = dir_name + f"/H{hurst}.png"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ts = np.linspace(0, 1, 1001)
    B = FBM(n=1000, hurst=hurst, length=1).fbm()
    
    plt.figure()
    plt.plot(ts, B, lw=2.5) #, ls="--", label='train data')
    plt.tight_layout()
    plt.savefig(file_name, dpi=500)
    

def main():
    plot_fBm(0.1)
    plot_fBm(0.25)
    plot_fBm(0.5)
    plot_fBm(0.75)
    plot_fBm(0.9)




if __name__=='__main__':
    main()
