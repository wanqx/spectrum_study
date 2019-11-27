'''
return pipe generator
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import re
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv
from load_basic import DATA_FOLDER, EXPER_FOLDER
EXPERFILES = [f'./{EXPER_FOLDER}/{x}' for x in os.listdir(f'./{EXPER_FOLDER}') if x.endswith('csv')]
EXPERFILES = sorted(EXPERFILES)
files = [x for x in os.listdir(DATA_FOLDER) if x.endswith("mod")]
PATTERN = re.compile("vtemp=(\d+) rtemp=(\d+)")
MAXTEMP = float(max([max(PATTERN.findall(x)[0]) for x in files]))
MINTEMP = float(min([min(PATTERN.findall(x)[0]) for x in files]))

def getFile(filename):
    tempx, tempy = [], []
    with open(filename) as f:
        fc = csv.reader(f)
        for i, item in enumerate(fc):
            if i == 0:
                continue
            wl, intense = item
            tempx.append(float(wl))
            tempy.append(float(intense))
        return tempx, tempy


def getData(v, r):
    resx, resy = [], []
    filename = f"oh vtemp={v} rtemp={r}.mod"
    with open(f"./{DATA_FOLDER}/{filename}", "r") as f:
        for item in f.readlines():
            item = item.rstrip()
            _x, _y = map(float, item.split(","))
            resx.append(_x)
            resy.append(_y)
    maxy = max(resy)
    resy = [each/maxy for each in resy.copy()]
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
    tempy = [each/maxy for each in tempy.copy()]
    tempx = [each*10 for each in tempx.copy()]
    return tempx, tempy

def ideal_data_generator():
    '''
    normalized
    '''
    length = len(files)
    for i, filename in enumerate(files):
        try:
            vtemp, rtemp = PATTERN.findall(filename)[0]
        except:
            print(filename)
        data = getData(v=vtemp, r=rtemp)[1]
        yield (int(vtemp)-MINTEMP)/(MAXTEMP-MINTEMP), (int(rtemp)-MINTEMP)/(MAXTEMP-MINTEMP), data

def exper_data_generator():
    length = len(EXPERFILES)
    for i, filename in enumerate(EXPERFILES):
        vtemp, rtemp = PATTERN.findall(filename)[0]
        data = getData(v=vtemp, r=rtemp)[1]
        yield int(vtemp), int(rtemp), data

DATA_FEATURES = len(getData(v=1000, r=1000)[1])
EXPER_FEATURES = len(getExperData(100)[1])
if __name__ == "__main__":
    #  print(getData(v=1000, r=1000))
    #  print(len(getData(v=1000, r=1000)))
    i = 0
    #  for batch in ideal_data_generator(100):
    #      i+=1
    #      print(batch[1])
    #      print(len(batch[1][1]))
    #      if i==2: break
    import tensorflow as tf
    df = tf.data.Dataset.from_generator(
        ideal_data_generator,
        (tf.float32, tf.float32, tf.float32),
        (tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None))
    )

    for value in df.shuffle(2).take(1).prefetch(tf.data.experimental.AUTOTUNE):
        print(value[2])
        print(value[1])
        i+=1
    print(i)

