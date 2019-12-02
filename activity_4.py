import numpy as np
import matrix as _matrix
import mathematic as _math
import pca
import plot
import pandas as pd
import kmeans

class Activity4:
    def __init__(self):
        pass

    def dataset_test(self, path, datatype="float"):        
        print(">>>>>> " + path + " <<<<<<")
        readmatrix = _matrix.ReadCsv(path, ";", datatype)
        readmatrix = _matrix.Transpose(readmatrix)   

        print("\n--------------------------------------------------")
    
    def Exercise1(self):
        data = [[1.9, 7.3], [3.4, 7.5], [2.5, 6.8], [1.5, 6.5], [3.5, 6.4], [2.2, 5.8], [3.4, 5.2], [3.6, 4], [5, 3.2], [4.5, 2.4], [6, 2.6], [1.9, 3], [1, 2.7], [1.9, 2.4], [0.8, 2], [1.6, 1.8], [1, 1]]
        _data = np.asarray(_matrix.AddColumn(_matrix.Copy(data), -1)) # create new column for classification
        x1 = np.asarray(_matrix.RemoveColumn(_data, len(_data[0])))
        y1 = _matrix.RemoveColumn(_data, 0)
        y1 = np.int_(np.transpose(_matrix.RemoveColumn(y1, 0))[0])
        plot.Kmeans2D(x1, y1, [[]], 'Plot pure data')

        newData, centroids = kmeans.Kmeans(data, 3)
        x = np.asarray(_matrix.RemoveColumn(newData, len(newData[0])))
        y = _matrix.RemoveColumn(newData, 0)
        y = np.int_(np.transpose(_matrix.RemoveColumn(y, 0))[0])        
        plot.Kmeans2D(x, y, centroids, 'Plot Kmeans')

    def test(self):
        self.Exercise1()
        # http://archive.ics.uci.edu/ml/datasets/OCT+data+%26+Color+Fundus+Images+of+Left+%26+Right+Eyes
        # http://archive.ics.uci.edu/ml/datasets/Perfume+Data

Activity4().test()