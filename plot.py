import matplotlib.pyplot as plt

def SimplePointData2D(data, title, labelx, labely):
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)
    plt.ylim(min(data[1]) - abs(min(data[1])*0.1), max(data[1]) + abs(max(data[1])*0.1))
    plt.plot(data[0], data[1], 'o', color='red')
    plt.show()

def Lda(X_lda, y, label_dict, title = 'Plot Lda'):
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()