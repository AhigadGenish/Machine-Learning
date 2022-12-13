# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import scipy.io as sio


# Ahigad Genish
# Clustering :

#get data :
def getData(path):
    data = sio.loadmat(path)
    train = []
    test = []
    for k in data.keys():
        if k.startswith('train'):
            train.append(data[k])
        elif k.startswith('test'):
            test.append(data[k])

    return train, test

#loss function:
def loss(m, c):
    loss = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            loss += np.linalg.norm(m[i][j] - c[i])
    return loss

#accuracy function:
def accuracy(c, test):
    counter = 0
    sum = 0
    for i in range(len(test)):
        for j in range(len(test[i])):
            sum += 1
            d = [0 for l in range(len(c))]
            for x in range(len(c)):
                d[x] = np.linalg.norm(test[i][j] - c[x])
            if d.index(min(d)) == i:
                counter += 1
    return (100 * counter) / sum

#calculate average:
def average(m):
    sum = [0 for i in range(len(m[0]))]
    for i in range(len(m)):
        for j in range(len(m[i])):
            sum[j] += m[i][j]
    for i in range(len(sum)):
        sum[i] /= len(m)

    return sum


def kMeans(k, data):
    c = []
    for i in range(k):
        c.append(data[(random.randint(0, len(data) - 1))])
    cBefore = c
    i = 0
    lossBefore = 0
    lossAfter = 0
    lossArr = []
    while True:
        m = [[] for e in range(k)]
        for j in tqdm(range(len(data))):
            d = [0 for q in range(len(c))]
            for r in range(len(c)):
                d[r] = np.linalg.norm(data[j] - c[r])
            m[d.index(min(d))].append(data[j])
        for v in range(len(c)):
            c[v] = average(m[v])
        cAfter = c
        i += 1
        if i == 1:
            lossBefore = loss(m, c)
            lossArr.append(lossBefore)
        else:
            lossAfter = loss(m, c)
            lossArr.append(lossAfter)
            if lossAfter - lossBefore >= 0 or lossBefore / lossAfter < 1.05:
                cAfter = cBefore
                break
            lossBefore = lossAfter
        cBefore = cAfter

    return lossArr, m, cAfter


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train, test = getData('mnist_all.mat')
    k = 10
    allTrain = []
    for i in train:
        for j in i:
            allTrain.append(j)
    loss, m, c = kMeans(k, allTrain)
    plt.plot(loss, ".r")
    plt.show()
    acc = accuracy(c, test)
    print("The success rate is :", acc, "%")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
