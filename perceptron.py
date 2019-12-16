import numpy as np
import matrix as _matrix
import mathematic as _math
import matplotlib.pyplot as plt

class Perceptron():
    
    def __init__(self, weightNumber, seasons, learningRate = 0.01):
        self.seasons = seasons
        self.learningRate = learningRate
        self.weights = np.zeros(weightNumber + 1) # cause of X0

    def train(self, data, y):
        for season in range(self.seasons):
            for dt, lab in zip(data, y):
                prediction = self.predict(dt)
                self.weights[1:] += self.learningRate * (lab - prediction) * dt
                self.weights[0] += self.learningRate * (lab - prediction)

    def predict(self, inputs):
        return 1 if (np.dot(inputs, self.weights[1:]) + self.weights[0]) >= 0 else 0