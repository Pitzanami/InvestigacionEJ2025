import numpy as np
from scipy.spatial import distance
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def indexToCoords(shape, index):
    _, Y, Z = shape
    
    x = index // (Y*Z)
    y = (index % (Y*Z))//Z
    z = index % Z
    
    return (x, y, z)

def coordsToIndex(shape, coords):
    _, Y, Z = shape
    x, y, z = coords
    
    return (x * (Y*Z)) + (y*Z + z)

def shapeToCoords(shape):
    l = []
    
    X, Y, Z = shape
    n = X*Y*Z
    
    for i in range(n):
        x = i // (Y*Z)
        y = (i % (Y*Z))//Z
        z = i % Z
        l.append((x,y,z))
    
    return l

def connProb(i, j, inh, shape):
    C = 0.3
    if i in inh and j in inh:
        C = 0.1
    if i in inh and j not in inh:
        C = 0.4
    if i not in inh and j in inh:
        C = 0.2
    
    a = indexToCoords(shape, i)
    b = indexToCoords(shape, j)
    D = distance.euclidean(a, b)
    l = 2

    return C * math.exp(- (D/l)**2)

def getConnMatrix(n, shape, perc = 0.2):
    inh = random.sample(range(n), int(n*perc))
    
    C_mat = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            umb = random.uniform(0, 1)
            C_mat[i][j] = 1 if connProb(i, j, inh, shape) < umb else 0
            if i in inh:
                C_mat[i][j] = C_mat[i][j] * -1
    
    return C_mat

def plotConnectivity(C, shape, show_arrows=False, show_labels=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n_nodes = C.shape[0]
    coords = np.array([indexToCoords(shape, i) for i in range(n_nodes)])
    
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='black', s=20, label='Neuronas')
    
    color_map = {1: 'blue', -1: 'red'}

    for i in range(n_nodes):
        for j in range(n_nodes):
            value = C[i, j]
            if value in color_map:
                start = coords[i]
                end = coords[j]
                if show_arrows:
                    # Dibujar flecha
                    direction = end - start
                    ax.quiver(*start, *direction, length=1, normalize=False, color=color_map[value], arrow_length_ratio=0.1, alpha=0.2)
                else:
                    # Dibujar línea simple
                    line = np.vstack([start, end])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color_map[value], alpha=0.2)
    
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Excitatoria'),
        Line2D([0], [0], color='red', lw=2, label='Inhibitoria')
        # Line2D([0], [0], color='green', lw=2, label='Sin conexión')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    if show_labels:
        for i, (x, y, z) in enumerate(coords):
            ax.text(x, y, z, f'{i}', color='black', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()