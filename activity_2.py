import numpy as np
import matrix as _matrix
import mathematic as _math
import pca
import matplotlib.pyplot as plt
import pandas as pd
import plot

class Activity2:
    def __init__(self):
        pass

    def dataset_test(self, path, datatype="float"):        
        print(">>>>>> " + path + " <<<<<<")
        originalMatrix = _matrix.ReadCsv(path, ";", datatype)
        originalMatrix = _matrix.Transpose(originalMatrix)
        auxMatrix = _matrix.Copy(originalMatrix)
        eValues, eVectors = pca.Pca(auxMatrix)
        
        _matrix.matrix_print("EigenVector", eVectors)
        _matrix.matrix_print("EigenValue", [eValues])

        # plot relevance components
        pca.PlotRelevance(eValues, "Relevance Components " + path)

        # plot data transformed in the new space
        plot.SimplePointData2D(originalMatrix, "Original " + path, "PC1", "PC2")

        # plot data transformed in the new space
        transformedData = pca.Transformation(originalMatrix, eVectors)
        plot.SimplePointData2D(transformedData, "Transformed " + path, "PC1", "PC2")

        print("\n--------------------------------------------------")
    
    def test(self):
        self.dataset_test("datasets/alpswater.csv")
        self.dataset_test("datasets/books_attend_grade.csv")
        self.dataset_test("datasets/us_census.csv")

Activity2().test()