import pandas as pd
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
def getdata():
    data = pd.read_csv("data.csv")
    return data

TOPIX = getdata()["TPX"] #.values
TOPIX.plot()
print(TOPIX)

#plt.plot(TOPIX, label = 'Airline Passangers Data')
plt.show()
