import numpy as np
import math

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
    D = np.linalg.norm(a - b)
    l = 2

    return C * math.exp(- (D/l)**2)

