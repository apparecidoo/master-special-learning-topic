import numpy as np
import matrix as _matrix
import mathematic as _math
import pca
import plot
import pandas as pd
import kmeans
from sklearn import datasets
from sklearn.decomposition import PCA

class Activity4:
    def __init__(self):
        pass

    def dataset_test(self, path, datatype="float"):        
        print(">>>>>> " + path + " <<<<<<")
        readmatrix = _matrix.ReadCsv(path, ";", datatype)

        _data = np.asarray(_matrix.AddColumn(_matrix.Copy(readmatrix), -1)) # create new column for classification
        x1, y1 = _matrix.DivideXY(_data)
        plot.Kmeans(x1, np.int_(y1), [[]], 'Plot pure data')

        newData, centroids = kmeans.Kmeans(readmatrix, 3)
        x, y = _matrix.DivideXY(newData)
        plot.Kmeans(x, np.int_(y), centroids, 'Plot Kmeans')

        print("\n--------------------------------------------------")
    
    def Exercise1(self):
        data = [[1.9, 7.3], [3.4, 7.5], [2.5, 6.8], [1.5, 6.5], [3.5, 6.4], [2.2, 5.8], [3.4, 5.2], [3.6, 4], [5, 3.2], [4.5, 2.4], [6, 2.6], [1.9, 3], [1, 2.7], [1.9, 2.4], [0.8, 2], [1.6, 1.8], [1, 1]]
        _data = np.asarray(_matrix.AddColumn(_matrix.Copy(data), -1)) # create new column for classification
        x1, y1 = _matrix.DivideXY(_data)
        plot.Kmeans(x1, np.int_(y1), [[]], path + ' - Plot pure data')

        newData, centroids = kmeans.Kmeans(data, 3)
        x, y = _matrix.DivideXY(newData)
        plot.Kmeans(x, np.int_(y), centroids, path + ' - Plot Kmeans')
    
    def Exercise2(self):
        self.dataset_test("datasets/books_attend_grade.csv")

    def Exercise3(self):
        iris = datasets.load_iris()
        x = iris.data

        # reduce dimensions with Pca
        resultPca = PCA(n_components = 2)
        resultPca.fit(x)
        PcaData = resultPca.transform(x)        

        _data = np.asarray(_matrix.AddColumn(_matrix.Copy(PcaData), -1)) # create new column for classification
        x1, y1 = _matrix.DivideXY(_data)
        plot.Kmeans(x1, np.int_(y1), [[]], 'Plot pure data')

        newData, centroids = kmeans.Kmeans(PcaData, 3)
        x, y = _matrix.DivideXY(newData)
        plot.Kmeans(x, np.int_(y), centroids, 'Plot Kmeans')

    def test(self):
        self.Exercise1()
        self.Exercise2()
        self.Exercise3()        

Activity4().test()