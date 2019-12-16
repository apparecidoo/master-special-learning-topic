import numpy as np
import matrix as _matrix
import mathematic as _math
import plot
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from perceptron import Perceptron

class Activity5:
    def __init__(self):
        pass
   
    def _BaseAndOrXor(self):
        data_train = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
        names = np.array(["True", "False"])
        return data_train, names
    
    def ExerciseAnd(self):
        data_train, names = self._BaseAndOrXor()        
        y = np.array([1, 0, 0, 0])

        percep = Perceptron(len(data_train[0]), 10)
        percep.train(data_train, y)
        result = []

        for i in data_train:
            result.append(percep.predict(i))
        result = np.asarray(result)
        plot.Perceptron(data_train, result, "Result 'AND'", names)

    def ExerciseOr(self):
        data_train, names = self._BaseAndOrXor() 
        y = np.array([0, 1, 1, 1])

        percep = Perceptron(len(data_train[0]), 10)
        percep.train(data_train, y)
        result = []

        for i in data_train:
            result.append(percep.predict(i))
        result = np.asarray(result)
        plot.Perceptron(data_train, result, "Result 'OR'", names)

    def ExerciseXor(self):
        data_train, names = self._BaseAndOrXor() 
        y = np.array([0, 1, 1, 0])

        percep = Perceptron(len(data_train[0]), 10)
        percep.train(data_train, y)
        result = []

        for i in data_train:
            result.append(percep.predict(i))
        result = np.asarray(result)
        plot.Perceptron(data_train, result, "Result 'XOR'", names)

    def ExerciseIris(self):
        iris = datasets.load_iris()
        names = iris.target_names
        data = iris.data
        target = iris.target

        # removing type 2
        data_train, y = _matrix.FilterByY(data, target, 2)

        percep = Perceptron(len(data_train[0]), 10)
        percep.train(data_train, y)
        result = []

        for i in data_train:
            result.append(percep.predict(i))
        
        plot.Perceptron(data_train, np.asarray(result), "Result Iris - Setosa x Versicolor", names)

    def test(self):
        self.ExerciseAnd()
        self.ExerciseOr()
        self.ExerciseXor()
        self.ExerciseIris()

Activity5().test()