from distutils.log import error
import math
import random
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
from scipy.stats import multivariate_normal

#Ahigad Genish
# Id 316228022

def gatData(path):
    data = sio.loadmat(path)
    train = []
    lableTrain = []
    test = []
    lableTest = []
    i = 0
    while (i < len(data['faces'])):
        for j in range(8):
            train.append(data['faces'][i+j])
            lableTrain.append(data['labeles'][i+j][0])
        i += 8
        for j in range(3):
            test.append(data['faces'][i+j])
            lableTest.append(data['labeles'][i+j][0])
        i += 3
    return np.array(train), np.array(lableTrain), np.array(test), np.array(lableTest)
# excpectation of the train:
def mean(train):
    mean = [0 for i in range(len(train[0]))]
    for i in range(len(train)):
        for j in range(len(train[i])):
            mean[j] += train[i][j]
    for i in range(len(mean)):
        mean[i] /= len(train)
    return np.array(mean)
# covariance of the train:

def cov(train,avg):
    cov = np.zeros((len(train[0]),len(train[0])))
    for i in range(len(train)):
        cov += np.dot((train[i]-avg).T,(train[i]-avg))
    for i in cov:
        for j in i:
            j /= len(train)
    return cov

def eigen(A):
    w ,v = np.linalg.eig(A)
    wv = []
    for i in range(len(w)):
        wv.append([w[i],v[i]])
    wv.sort(key = lambda x: x[0],reverse=True)
    return wv

def PCA(wv,k,average,train):
    if(k > len(train[0]) or k <= 0):
        return train

    V = np.array([wv[i][1] for i in range(len(wv))])
    for x in train:
        np.subtract(x,average)
    Newtrain = np.dot(train, V[:,:k])
    return Newtrain

def distance(x,y):
    return np.linalg.norm(x-y)

def KNN(train,lableTrain,test,lableTest):
    predictadeLableTest=[]
    for x in test:
        ans=[(distance(x,i)) for i in train]
        predictadeLableTest.append(lableTrain[np.argmin(ans)])
    return predictadeLableTest

    


def main():
    print ("PCA")
    path='C:\\Users\\USER\\PycharmProjects\\PCA\\facesData.mat'
    train, lableTrain, test, lableTest = gatData(path)
    average = mean(train)
    covMat = cov(train, average)
    wv = eigen(covMat)

    accuracy =[]
    for k in (tqdm(range(1, len(train[0])+1))):
        Newtrain = PCA(wv,k, average,train)
        Newtest = PCA(wv,k,average,test)
        predictadeLableTest = KNN(Newtrain,lableTrain, Newtest, lableTest)
        sum = 0
        for i in range(len(predictadeLableTest)):
            if(predictadeLableTest[i]==lableTest[i]):
                sum += 1
        accuracy.append(sum/len(predictadeLableTest))
    plt.plot(accuracy)
    plt.show()


    '''evg=mean(train)
    evg=evg.reshape(32,32).T
    plt.imshow(evg,cmap="gray")
    plt.show()'''

    
if __name__ == "__main__":
    main()