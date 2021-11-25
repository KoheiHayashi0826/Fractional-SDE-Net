import pandas as pd

def getdata():
    TOPIX = pd.read_csv("data.csv")
    #print(TOPIX)
    return TOPIX

T = getdata()
print(T)
