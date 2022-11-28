
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import openpyxl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
def sig(x):
    return 1/(1 + np.exp(-x))

def binClassifier(path_train1, path_train2, iteration):
    # get data:

    train1 = pd.read_excel(path_train1).to_numpy()
    train2 = pd.read_excel(path_train2).to_numpy()
    #ex1 = train1[0]
    #ex1 = np.reshape(ex1, (28, 28))
    #image = plt.imshow(ex1)
    #plt.show()
    #ex2 = train2[0]
    #ex2 = np.reshape(ex2, (28, 28))
    #image = plt.imshow(ex2)
    #plt.show()

    N = len(train1) + len(train2)
    Wi = np.zeros((1, len(train1[0])))  # 1Xd
    for i in range(len(Wi)):
        Wi[i] = 0.001

    epsilon = 0.0001
    e = epsilon
    epsilondivN = epsilon / N
    run = iteration
    L = []
    iter = [i for i in range(run)]

    for j in tqdm(range(run)):

        # Derivative
        der = np.zeros((1, len(train1[0])))  # 1Xd
        Xt = train1  # nXd
        WiXt = np.dot(Xt, Wi.T)  # nXdXdX1 = #nX1
        sigm1 = sig(WiXt)  # nX1
        temp = 0 - sigm1  # nX1
        dif = np.dot(temp.T, Xt)  # 1XnXnXd = 1Xd
        der += dif

        Xt = train2  # nXd
        WiXt = np.dot(Xt, Wi.T)
        sigm2 = sig(WiXt)
        temp = 1 - sigm2
        dif = np.dot(temp.T, Xt)
        der += dif
        Wi += epsilondivN * der
        sum1 = 0
        sum2 = 0
        for i in range(len(sigm1)):
            sum1 += np.log(1 - sigm1[i] + e)
        for i in range(len(sigm2)):
            sum2 += np.log(sigm2[i] + e)

        L.append((sum1 + sum2) / N)

    return L, iter, Wi


def succssesRate(Wi, path_test1, path_test2):


    test1 = pd.read_excel(path_test1).to_numpy()
    test2 = pd.read_excel(path_test2).to_numpy()
    N = len(test1) + len(test2)
    Xt = test1
    XtWi = np.dot(Xt, Wi.T)
    sigOne = sig(XtWi)
    success = 0
    for s in sigOne:
        if s <= 0.5:
            success += 1

    Xt = test2
    XtWi = np.dot(Xt, Wi.T)
    sigTwo = sig(XtWi)
    for s in sigTwo:
        if s >= 0.5:
            success += 1

    succssesRate = 100*(success / N)

    return succssesRate



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    numOfIteration = 100
    ans, iter, Wi = binClassifier('train1.xlsx', 'train2.xlsx',numOfIteration)
    plt.plot(iter, ans)
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title('Loss function as a function of the number of iterations')
    plt.show()
    sR = succssesRate(Wi, 'test1.xlsx', 'test2.xlsx')
    print('The successRate is: ', sR ,'%')





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
