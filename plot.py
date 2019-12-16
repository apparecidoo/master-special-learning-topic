import matplotlib.pyplot as mplt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits import mplot3d

def SimplePointData2D(data, title, labelx, labely):
    mplt.xlabel(labelx)
    mplt.ylabel(labely)
    mplt.title(title)
    mplt.ylim(min(data[1]) - abs(min(data[1])*0.1), max(data[1]) + abs(max(data[1])*0.1))
    mplt.plot(data[0], data[1], 'o', color='red')
    mplt.show()

def LdaIris2D(X_lda, y, label_dict, title = 'Plot Lda'):
    ax = mplt.subplot(111)
    for label,marker,color in zip(
        range(0,3),('^', 's', 'o'),('blue', 'red', 'green')):
            mplt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
            )

    mplt.xlabel('LD1')
    mplt.ylabel('LD2')

    leg = mplt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    mplt.title(title)

    # hide axis ticks
    mplt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    mplt.grid()
    mplt.tight_layout
    mplt.show()

def Kmeans(x, y, centroids = [[]], title = 'Kmeans'):
    if(len(x[0]) == 2): # 2 dimensions
        mplt.title(title)
        mplt.scatter(x[:, 0], x[:, 1], s=50)    
        mplt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='viridis')
        
        if(not centroids == [[]]):
            mplt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.4)

        mplt.show()
    else:
        if(len(x[0]) == 3):
            mplt.title(title)
            fig = mplt.figure(title, figsize=(5,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:,0], x[:,1], x[:,2], c=y , cmap='viridis', s=50)
            if(not centroids == [[]]):
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', s=100, alpha=0.4)
            mplt.show()
        else:
            print('Cannot plot with 4 or more dimensions')

def Perceptron(data, labels, title, names=["",""]):
    colors = ['r', 'b', 'g']
    markers = ['x', 'o', '^']
    
    mplt.title(title)
    mplt.xlabel("X")
    mplt.ylabel("Y")
    for i in range(len(data)):
        mplt.scatter(data[i][0], data[i][1], s=100, c=colors[labels[i]], marker=markers[labels[i]])
    
    patch1 = mpatches.Patch(color='red', label=names[0])
    patch2 = mpatches.Patch(color='blue', label=names[1])
    mplt.legend(handles=[patch1, patch2])
    mplt.show()