
import matplotlib.pyplot as plt
import numpy as np
import random
import math
#Ahigad Genish
def createX(row,col):

    X = np.random.rand(row, col)
    X_T = X.T
    B = np.random.rand(col, 1)
    for i in range(0, len(B)):
        B[i] = 1000
    Ytag = np.dot(X,B)
    Z = np.dot(X_T,X)
    Zinv = np.linalg.inv(Z)
    Btag = np.dot(Zinv, X_T)
    return X, Ytag, Btag, B

def linearReg(X, Ytag, Btag, B, mu, sigma):

    variance = sigma
    expectation = mu

    e = np.random.normal(expectation, variance, len(X))
    E = np.random.rand(len(X), 1)
    for i in range(len(E)):
        E[i] = e[i]

    #Ytag = X*B
    # Y = Ytag + E
    # X_T = X.T
    Y = Ytag + E
    #Btg = (X_T*X)^-1 * X_T
    #B^ = Z^-1 * X_T*Y
    Bhat = np.dot(Btag, Y)
    error = np.dot(abs(B-Bhat).T, abs(B-Bhat))
    return error





if __name__ == '__main__':

    #create matrix X(nxd) represent number of examples on numbers of features.
    X, Ytag, Btag, B = createX(1500, 600)
    #define variance and expectation of the noise
    tests = 100
    mu = 0
    error = []
    variance = []
    for i in range(tests):
        sigma = i//2
        variance.append(sigma)
        error.append(linearReg(X, Ytag,Btag, B, mu, sigma)[0][0])

    plt.plot(variance, error,'r.')
    plt.show()

