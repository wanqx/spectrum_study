import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

files = [x for x in os.listdir("./data") if x.endswith("mod")]


def getData(v, r):
    resx, resy = [], []
    filename = f"oh vtemp={v} rtemp={r}.mod"
    with open(f"./data/{filename}", "r") as f:
        for item in f.readlines():
            item = item.rstrip()
            _x, _y = map(float, item.split(","))
            resx.append(_x)
            resy.append(_y)
    return resx, resy

def plot(v, r, c=None):
    plt.plot(*getData(v, r), label=f"vt={v} rt={r}", c=c)

if __name__ == "__main__":
    plot(v=4990, r=4990, c="crimson")
    plt.legend()
    plt.show()
