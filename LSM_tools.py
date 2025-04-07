import numpy as np
from scipy.spatial import distance
import math
import random

def indextocoords(shape, index):
    _, Y, Z = shape
    
    x = index // (Y*Z)
    y = (index % (Y*Z))//Z
    z = index % Z
    
    return (x, y, z)

def coordstoindex(shape, coords):
    _, Y, Z = shape
    x, y, z = coords
    
    return (x * (Y*Z)) + (y*Z + z)

def shapetocoords(shape):
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
    
    a = indextocoords(shape, i)
    b = indextocoords(shape, j)
    D = distance.euclidean(a, b)
    l = 2

    return C * math.exp(- (D/l)**2)

def getConnMatrix(n, shape, perc = 0.2):
    inh = random.sample(range(n), int(n*perc))
    
    C_mat = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            umb = random.uniform(0, 1)
            C_mat[i][j] = 1 if connProb(i, j, inh, shape) > umb else 0
            if i in inh:
                C_mat[i][j] = C_mat[i][j] * -1
    
    return C_mat