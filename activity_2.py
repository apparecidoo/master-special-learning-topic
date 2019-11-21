import numpy as np
import matrix as _matrix
import mathematic as _math
import pca
import matplotlib.pyplot as plt
import pandas as pd

class Activity2:
    def __init__(self):
        pass

    def dataset_test(self, path, datatype="float"):        
        print(">>>>>> " + path + " <<<<<<")
        readmatrix = _matrix.ReadCsv(path, ";", datatype)
        readmatrix = _matrix.Transpose(readmatrix)

        eValues, eVectors = pca.Pca(readmatrix)
        
        _matrix.matrix_print("EigenVector", eVectors)
        _matrix.matrix_print("EigenValue", [eValues])

        # plot of relevance components
        left = [i for i in range(len(eValues))]
        tick_label = ["PC" + str(i+1) for i in range(len(eValues))]
        right = [i*100 / sum(eValues) for i in eValues]
        rect = plt.bar(left, right, tick_label = tick_label, width = 0.5, label = right)
        plt.yticks(np.arange(0, 100+1, step=20))
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width()/2.0, height, '%f %%' % float(height), ha='center', va='bottom')
        plt.title("Relevance Components")
        plt.show()

        # plot of data transformed in the new space
        transformedData = pca.PcaTransformation(readmatrix, eVectors)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(path)
        plt.ylim(min(transformedData[1]) - abs(min(transformedData[1])*0.1), max(transformedData[1]) + abs(max(transformedData[1])*0.1))
        plt.plot(transformedData[0], transformedData[1], 'o', color='red')
        plt.show()

        print("\n--------------------------------------------------")
    
    def test(self):
        self.dataset_test("datasets/alpswater.csv")
        self.dataset_test("datasets/books_attend_grade.csv")
        self.dataset_test("datasets/us_census.csv")

Activity2().test()