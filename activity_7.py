import keras
import numpy as np
from lenet import Lenet

class Activity7:
    def __init__(self):
        pass

    def test(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        lenet = Lenet()
        lenet.Fit(x_train, y_train, 10, 128)
        score = lenet.Evaluate(x_test, y_test)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = lenet.Predict(x_test)

        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        print()
        print(lenet.ConfusionMatrix(y_test, y_pred))

Activity7().test()