import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from read_data import getExperData, getData, IDEALMEAN, EXPERFILES
from load_basic import NET_NAME, DATA_SIZE
from conv1D import ConvNet


files = [x for x in os.listdir("./experData") if x.endswith("mod")]



def plot(index, c=None, silent=False):
    x, y = getExperData(EXPERFILES[index])
    #  for i, each in enumerate(y):
    #      if i>600: y[i]=0.
    y1 = y + [0.]*(1313-len(y))
    maxy = max(y1)
    y1 = [100*eachy/maxy for eachy in y1.copy()]
    if not silent:
        plt.plot(x, y, label=f"index={index}", c=c)
        plt.show()
        plt.plot(y1, label="net input")
        plt.legend()
        plt.show()
    return y1

def idealplot(v, r, c=None):
    plt.plot(*getData(v, r), label=f"v={v}, r={r}", c=c)



if __name__ == "__main__":
    new_model = ConvNet()
    new_model.load_weights(NET_NAME)
    data = plot(222, c="crimson")
    idealplot(v=3384, r=3384, c="crimson")
    plt.legend()
    plt.show()

    pred = new_model(data).numpy()[0][0]
    pred += IDEALMEAN
    print(pred)
    PRED = []
    for i in range(1, len(EXPERFILES)):
        data = plot(i, silent=True)
        pred = new_model(data).numpy()[0][0]
        pred += IDEALMEAN
        if pred < 100 or pred > 6000: pred = 297
        PRED.append(pred)
    plt.plot(PRED, c="crimson")
    plt.ylabel("T / K")
    plt.ylim(0, 4000)
    plt.show()

