import numpy as np
import matrix as _matrix
import mathematic as _math
import lda as lda
import plot as plot
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA

class Activity3:
    def __init__(self):
        pass
    
    def test(self):
        # import some data to play with
        iris = datasets.load_iris()
        names = iris.target_names
        x = iris.data
        y = iris.target

        # Lda without PCA
        resultLda = lda.Lda(x, y)
        LdaData = lda.Transform(x, resultLda, 2)
        plot.Lda(LdaData, y, names, 'LDA without PCA: Iris projection with first 2 linear discriminants')

        # Lda with Pca
        resultPca = PCA(n_components = 2)
        resultPca.fit(x)
        PcaData = resultPca.transform(x)
        resultLda2 = lda.Lda(PcaData, y)
        print(resultLda2)
        LdaData = lda.Transform(PcaData, resultLda2, 2)
        plot.Lda(LdaData, y, names, 'LDA with PCA: Iris projection with first 2 linear discriminants')

Activity3().test()