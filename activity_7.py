import numpy as np
import matrix as _matrix
import mathematic as _math
import plot
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from multilayer_perceptron import MultilayerPerceptron

class Activity7:
    def __init__(self):
        pass
   
    def ExerciseTest(self):       
        dataset = [[2.7810836,2.550537003,0],
            [1.465489372,2.362125076,0],
            [3.396561688,4.400293529,0],
            [1.38807019,1.850220317,0],
            [3.06407232,3.005305973,0],
            [7.627531214,2.759262235,1],
            [5.332441248,2.088626775,1],
            [6.922596716,1.77106367,1],
            [8.675418651,-0.242068655,1],
            [7.673756466,3.508563011,1]]
        
        names = np.array(["True", "False"])
        n_inputs = len(dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in dataset]))

        mlp = MultilayerPerceptron(n_inputs, n_outputs, 2)
        mlp.Train(dataset)
        result = []

        for row in dataset:
            prediction = mlp.Predict(row)
            result.append(prediction)
            print('Expected=%d, Got=%d' % (row[-1], prediction))

        plot.Perceptron(dataset, np.asarray(result), "Result 'TEST'", names)

    def ExerciseXor(self):
        data_train = np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        names = np.array(["True", "False"])
        
        n_inputs = len(data_train[0]) - 1
        n_outputs = len(set([row[-1] for row in data_train]))
        print(n_outputs)
        print(n_inputs)
        mlp = MultilayerPerceptron(n_inputs, n_outputs, 2, 3)
        mlp.Train(data_train, 0.1, 1000)
        result = []
        
        for row in data_train:
            prediction = mlp.Predict(row)
            result.append(prediction)
            print('Expected=%d, Got=%d' % (row[-1], prediction))

        plot.Perceptron(data_train, np.asarray(result), "Result 'XOR'", names)
        
    def ExerciseIris(self):
        iris = datasets.load_iris()
        names = iris.target_names
        dataTransp = iris.data.T
        target = iris.target
        data = _matrix.Copy(dataTransp)
        data.append(target)
        data_train = np.transpose(data)
        newData = _matrix.Create(len(data_train), len(data_train[0]), 0)
        
        # adaptation to remove float of label
        for i in range(len(data_train)):
            for j in range(len(data_train[0])):
                if(j == len(data_train[0]) - 1):
                    newData[i][j] = int(data_train[i][j])
                else:
                    newData[i][j] = data_train[i][j]

        n_inputs = len(newData[0]) - 1
        n_outputs = len(set([row[-1] for row in newData]))
        
        mlp = MultilayerPerceptron(n_inputs, n_outputs, 3)
        mlp.Train(newData)
        result = []

        for row in newData:
            prediction = mlp.Predict(row)
            result.append(prediction)
            print('Expected=%d, Got=%d' % (row[-1], prediction))

        #plot.Perceptron(newData, np.asarray(result), "Result Iris - Setosa x Versicolor", names)

    def test(self):
        # self.ExerciseTest()
        self.ExerciseXor()
        # self.ExerciseIris()

Activity7().test()