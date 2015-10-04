# PerceptronVisualizer.py
# Visualize perceptron learning algorithm (PLA) on 2D data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import random
import sys

# sample data for Blue and Red classes
dataClassBlue = [(1,2), (3,5), (7,3), (2,8), (1,5)]
dataClassRed = [(8,12), (11,8), (3,13), (6,10), (13,2)]

BlueX1 = np.array([i[0] for i in dataClassBlue])
BlueX2 = np.array([i[1] for i in dataClassBlue])
BlueY  = np.ones(5)
Blue = np.array([np.ones(5),BlueX1, BlueX2]).transpose()

RedX1 = np.array([i[0] for i in dataClassRed])
RedX2 = np.array([i[1] for i in dataClassRed])
RedY  = np.zeros(5)-1
Red = np.array([np.ones(5),RedX1, RedX2]).transpose()

BlueRedX = np.vstack((Blue, Red)).transpose()
BlueRedY = np.concatenate([BlueY,RedY])
xLine = np.linspace(0,15,151)
weightsInitial = np.array([1.3,-1.4,1.7])

def mis(BlueRedX, BlueRedY, w):
    misclassified = np.sign(w.dot(BlueRedX)) == BlueRedY
    return np.where(misclassified == False)[0]

def plotCurrent(yLine, currentPoint = [], currentLabel = 1, allMis = [], allMisLabel = []):
    plt.close('all')
    plt.plot(BlueX1,BlueX2, 'o', color = 'b')
    plt.plot(RedX1,RedX2, 'o', color = 'r')
    color = {1:'b',-1:'r'}
    for j, p in enumerate(zip(allMis, allMisLabel)):
        if j == 0: m, = plt.plot(p[0][1], p[0][2], 'o', color='g', markersize=25, label='misclassified')
        else: plt.plot(p[0][1], p[0][2], 'o', color='g', markersize=25)
        plt.plot(p[0][1], p[0][2], 'o', color=color[p[1]], markersize=5)
    if len(currentPoint) != 0:
        c, = plt.plot(currentPoint[1], currentPoint[2], 'o', color='orange', markersize=25, label="current point")
        plt.plot(currentPoint[1], currentPoint[2], 'o', color=color[currentLabel], markersize=5)

    plt.plot(xLine, yLine, color='k', linewidth=2)
    if -w[1]/w[2] > 0 and -w[0]/w[2] >= 15: blue, red = 0, 15
    else: blue, red = 15, 0
    plt.fill_between(xLine, yLine, y2=blue, alpha = .3, color='b')
    plt.fill_between(xLine, yLine, y2=red,  alpha = .3, color='r')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim([0,15])
    plt.ylim([0,15])
    if len(currentPoint) != 0:
        plt.legend(handler_map={m: HandlerLine2D(numpoints=1), c: HandlerLine2D(numpoints=1)},
                borderpad=0.8, labelspacing=1.2, loc='upper right')
    plt.title("Perceptron Visualizer")
    plt.show(block=False)

def printParams():
    print "i = {}\t\tw = {}\t\tlenMis = {}\tmis = {}".format(i, w, len(misclassified), str(misclassified))

for i in range(300):
    if i == 0: w = weightsInitial
    yLine = -w[1]/w[2]*xLine - w[0]/w[2]
    misclassified = mis(BlueRedX, BlueRedY, w)
    if len(misclassified) == 0: break
    else:
        index = random.choice(misclassified)
        currentPoint = BlueRedX.transpose()[index]
        currentLabel = BlueRedY[index]

    if len(sys.argv) > 1 and sys.argv[1] == '1':
        printParams()
        plotCurrent(yLine, currentPoint, currentLabel,
            BlueRedX.transpose()[misclassified], BlueRedY[misclassified])
        plt.waitforbuttonpress()

    # update the weights with a misclassified point
    w = w + currentLabel * currentPoint

plotCurrent(yLine)
printParams()