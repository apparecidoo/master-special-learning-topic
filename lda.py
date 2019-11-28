import numpy as np
import matrix as _matrix
import mathematic as _math
from sklearn import datasets
from collections import OrderedDict

def MeanVectors(x, y, numberClass = None):
    if(numberClass == None):
        numberClass = len(list(OrderedDict.fromkeys(y)))

    np.set_printoptions(precision=4)
    meanVectors = []
    for cl in range(0, numberClass):
        meanVectors.append(np.mean(x[y==cl], axis=0))
    
    return meanVectors

def WithInClass(x, y, meanVec):
    matrixWithin = np.zeros((len(x[0]),len(x[0])))
    for cl,mv in zip(range(0,len(x[0])), meanVec):
        class_sc_mat = np.zeros((len(x[0]),len(x[0])))
        for row in x[y == cl]:
            row, mv = row.reshape(len(x[0]),1), mv.reshape(len(x[0]),1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        matrixWithin += class_sc_mat
    
    return matrixWithin

def BetweenClass(x, y, meanVec):
    overallMean = np.mean(x, axis=0)
    betClass = np.zeros((len(x[0]), len(x[0])))
    for i, meanVec in enumerate(meanVec):  
        n = x[y==i,:].shape[0]
        meanVec = meanVec.reshape(len(x[0]),1)
        overallMean = overallMean.reshape(len(x[0]), 1)
        betClass += n * (meanVec - overallMean).dot((meanVec - overallMean).T)

    return betClass

def ExplainedVariance(eigenPairs, eigenVal):
    print('Variance explained:\n')
    eigenValSum = sum(eigenPairs)
    for i,j in enumerate(eigenPairs):
        print('Eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigenValSum).real))

def Transform(data, eigPair, nComponents):
    W = []
    for i in range(nComponents):
        W.append(eigPair[i][1])
    W = np.transpose(W)
    
    return data.dot(W)

def Lda(x, y, numberClass = None):
    if(numberClass == None):
        numberClass = len(list(OrderedDict.fromkeys(y)))
    
    meanVec = MeanVectors(x, y, numberClass)
    matrixWithin = WithInClass(x, y, meanVec)
    matrixBetween = BetweenClass(x, y, meanVec)
    
    # calculating eigen values and vectors
    eigenVal, eigenVec = np.linalg.eig(np.linalg.inv(matrixWithin).dot(matrixBetween))
    
    # Make a list sorted of (eigenvalue, eigenvector) tuples
    eigenPairs = [(np.abs(eigenVal[i]), eigenVec[:,i]) for i in range(len(eigenVal))]
    eigenPairs = sorted(eigenPairs, key=lambda k: k[0], reverse=True)

    return eigenPairs