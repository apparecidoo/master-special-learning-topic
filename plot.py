"""
Created on Wed Nov 1 19:03:56 2019
@author: Murillo Freitas Bouzon
@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)
Funções para exibição de gráficos
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ExplainedVariance(eValues):
    left = [i for i in range(len(eValues))]
    tick_label = ["PC" + str(i+1) for i in range(len(eValues))]
    right = [i*100 / sum(eValues) for i in eValues]
    rect = plt.bar(left, right, tick_label = tick_label, width = 0.5, label = right)
    plt.yticks(np.arange(0, 100+1, step=20))
    for r in rect:
        height = r.get_height()
        plt.text(r.get_x() + r.get_width()/2.0, height, '%f %%' % float(height), ha='center', va='bottom')
    plt.title("Variância Explicada")
    plt.show()

def TransformedData(transData, title):
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.ylim(min(transData[1]) - abs(min(transData[1])*0.1), max(transData[1]) + abs(max(transData[1])*0.1))
    plt.plot(transData[0], transData[1], 'o', color='red')
    plt.show()
