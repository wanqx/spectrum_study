import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv
EXPERFILES = [f'./experData/{x}' for x in os.listdir('./experData') if x.endswith('csv')]
EXPERFILES = sorted(EXPERFILES)

def getFile(file):
    tempx, tempy = [], []
    with open(file) as f:
        fc = csv.reader(f)
        for i, item in enumerate(fc):
            if i == 0:
                continue
            wl, intense = item
            tempx.append(float(wl))
            tempy.append(float(intense))
        return tempx, tempy

FILENAME = "./等温度"
files = [x for x in os.listdir(FILENAME) if x.endswith("mod")]

def getData(v, r):
    resx, resy = [], []
    filename = f"oh vtemp={v} rtemp={r}.mod"
    with open(f"./{FILENAME}/{filename}", "r") as f:
        for item in f.readlines():
            item = item.rstrip()
            _x, _y = map(float, item.split(","))
            resx.append(_x)
            resy.append(_y)
    return resx, resy

def getExperData(index):
    tempx, tempy = [], []
    with open(EXPERFILES[index]) as f:
        fc = csv.reader(f)
        for i, item in enumerate(fc):
            if i == 0:
                continue
            wl, intense = item
            tempx.append(float(wl))
            tempy.append(float(intense))
    #  tempy = signal.detrend(tempy)
    maxy = max(tempy)
    tempy = [each/maxy*100 for each in tempy.copy()]
    tempx = [each*10 for each in tempx.copy()]
    return tempx, tempy

def plot(v, r, c=None, fnc=getData):
    try:
        wavelength, intensity = fnc(v=v, r=r)
    except TypeError:
        wavelength, intensity = fnc(v)
    print(fnc.__name__, ": \t", len(wavelength))
    #  N = 1e5
    #  amp = 2*np.sqrt(2)
    #  freq = 1234.0
    #  noise_power = 0.001 * fs / 2
    #  time = np.arange(N) / fs
    #  x = amp*np.sin(2*np.pi*freq*time)
    #  x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, Pxx_den = signal.welch(intensity, len(wavelength)*4, nperseg=len(wavelength), average="median")
    plt.semilogy(f, Pxx_den, label=f"v = {v} r = {r}")

def plotOrigin(v, r, c=None, fnc=getData):
    try:
        wavelength, intensity = fnc(v=v, r=r)
    except TypeError:
        wavelength, intensity = fnc(v)
    plt.plot(wavelength, intensity, label=f"v = {v} r = {r}")

for wave in (3000,"OHxx"):
    plot(v=wave, r=wave)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.show()

#  for index in (141, 151, 161, 171):
#      plot(v=index, r=index, fnc=getExperData)
#  plt.xlabel('frequency [Hz]')
#  plt.ylabel('PSD [V**2/Hz]')
#  plt.legend()
#  plt.show()

#  for index in (141, 151, 161, 171):
#      plotOrigin(v=index, r=index, fnc=getExperData)
#  plt.xlabel('frequency [Hz]')
#  plt.ylabel('PSD [V**2/Hz]')
#  plt.legend()
#  plt.show()

for wave in (3000, "OHxx"):
    plotOrigin(v=wave, r=wave)
plt.xlabel('wavelength')
plt.ylabel('intensity')
plt.legend()
plt.show()
