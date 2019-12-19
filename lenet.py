import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import numpy as np

class Lenet:
    def __init__(self):
        self.lenet = self.__InitializeLenet()

    def __InitializeLenet(self):
        lenet = keras.models.Sequential()

        lenet.add(keras.layers.Conv1D(
            filters=6,
            kernel_size=5,
            activation='relu',
            use_bias=True,
            input_shape=(28, 28)))
        lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

        lenet.add(keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu', use_bias=True))
        lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

        lenet.add(keras.layers.Flatten())

        lenet.add(keras.layers.Dense(120, activation='relu', use_bias=True))
        lenet.add(keras.layers.Dense(84, activation='relu', use_bias=True))
        lenet.add(keras.layers.Dense(10, activation='softmax', use_bias=True))

        lenet.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy']
        )

        return lenet

    def Fit(self, x_train, y_train, epochs = 10, batch_size = 128):
        self.lenet.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    def Predict(self,x_test):
        return self.lenet.predict(x_test)

    def Evaluate(self, x_test, y_test):
        return self.lenet.evaluate(x_test, y_test, verbose=0)

    def ConfusionMatrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)