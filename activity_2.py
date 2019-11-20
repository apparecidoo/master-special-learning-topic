import numpy as np
import matrix as _matrix
import mathematic as _math
import pca
import plot
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

        plot.ExplainedVariance(eValues)

        transData = pca.PcaTransform(readmatrix, eVectors)
        plot.TransformedData(transData, path)

        print("\n--------------------------------------------------")
    
    def test(self):
        self.dataset_test("datasets/alpswater.csv")
        self.dataset_test("datasets/books_attend_grade.csv")
        self.dataset_test("datasets/us_census.csv")

Activity2().test()