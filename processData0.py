import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate
import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from read_data import getExperData, getData, IDEALMEAN
from load_basic import NET_NAME, DATA_SIZE
from conv1D import ConvNet


def _getRawExperData(filename):
    tempx, tempy = [], []
    with open(filename, "r") as f:
        fc = csv.reader(f)
        for i, item in enumerate(fc):
            if i == 0:
                continue
            wl, intense = item
            tempx.append(float(wl))
            tempy.append(float(intense))
    #  tempy = signal.detrend(tempy)
    maxy = max(tempy)
    tempy = [100*each/maxy for each in tempy.copy()]
    return tempx, tempy

def getExperData(filename):
    x, y = _getRawExperData(filename)
    x = np.array(x)
    y = np.array(y)
    xx = np.linspace(x.min(), x.max(), 1313)
    f = interp1d(x, y, kind = "cubic")
    return f(xx)

if __name__ == "__main__":
    new_model = ConvNet()
    new_model.load_weights(NET_NAME)
    data = getExperData("test.mod")
    #  print(data)
    pred = new_model(data).numpy()[0][0]
    pred += IDEALMEAN
    print(pred)

    plt.plot(data)
    plt.show()
