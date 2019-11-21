import numpy as np
import matplotlib.pyplot as plt

def Simple2D(data, title, labelx, labely):
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.ylim(min(data[1]) - abs(min(data[1])*0.1), max(data[1]) + abs(max(data[1])*0.1))
    plt.plot(data[0], data[1], 'o', color='red')
    plt.show()
