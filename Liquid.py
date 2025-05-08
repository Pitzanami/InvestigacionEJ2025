import numpy as np
from scipy.spatial import distance
import math
import random

import numpy as np

class controller:
    def __init__(self, K:int, W, V_0, gamma = 0.5, i_ext = 0.1, theta = 1):
        self.N = len(W)
        self.gamma = gamma
        self.i_ext = i_ext
        self.theta = theta
        self.W = W
        self.V_0 = V_0              # Estado inicial
        self.K = K                  
        self.V = np.concatenate((V_0, np.zeros((self.N, K))), axis = 1)  
        self.Z = self.V.copy()
        
        for i in range(self.N):
            self.Z[i][0] = 1 if self.V[i][0] >= theta else 0
        
    def cambio(self, i, k):
        acum = sum([self.W[i][j]*self.Z[j][k-1] for j in range(self.N)])

        self.V[i][k] = self.gamma * self.V[i][k-1] * (1 - self.Z[i][k-1]) + self.i_ext + acum
        self.Z[i][k] = 1 if self.V[i][k] >= self.theta else 0 

    def print_V(self):
        print(self.V)

    def simulacion(self):
        for k in range(1, self.K):
            for i in range(self.N):
                self.cambio(i, k)
        return self.Z
    
    def simu(self, inputs):
        for k in range(1, self.K):
            for i in range(self.N):
                self.cambio(i, k)
        return self.Z
    
# ----------------------------------------------------------------------

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

def connProb(i, j, inh, shape, l):
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

    return C * math.exp(- (D/l)**2)

# CLASS LIQUID

class Liquid:
    def __init__(self, shape_liquid: tuple, n_channels:int, t:int, 
                 l:float=2.2, prob_IL:float=0.3, perc_inh:float = 0.2):

        # hiperparametros 
        self.l = l
        self.t = t
        self.prob_IL = prob_IL
        self.perc_inh = perc_inh
        
        self.shape_liquid = shape_liquid
        self.n_neurons = shape_liquid[0] * shape_liquid[1] * shape_liquid[2]
        
        self.inh, self.C_eq = self.__getConnMatrix()

        self.n_channels = n_channels
        
        self.input = self.__getInputLayer()

        W_random = np.random.uniform(0, 1, (self.n_neurons, self.n_neurons+self.n_channels))
        self.C_eq = np.hstack([self.C_eq, self.input])
        
        self.WC = W_random * self.C_eq

        # W_random = np.random.uniform(0, 1, (self.n_neurons, self.n_neurons))
        # self.WC = W_random * self.C_eq
        
        self.Z_0 = np.zeros((self.n_neurons, 1))

        # Controller
        self.cpg = controller(self.t, self.WC, self.Z_0)

    def simulacion(self, inputs):
        self.sim = self.cpg.simu(inputs)
        return self.sim
    
    def photo(self, factor:float):
        n_frames = factor*self.t
        
        sample = self.sim[-n_frames:]
        photo = []
        
        for s in sample:
            for i in s:
                photo.append(np.sum(i)/n_frames)
        
        return photo
        
    
    def __getConnMatrix(self):
        inh = np.random.choice(range(self.n_neurons), int(self.n_neurons * self.perc_inh), replace=False)
        
        C_mat = np.zeros((self.n_neurons, self.n_neurons))
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                
                umb = np.random.uniform(0, 1)
                if umb < connProb(i, j, inh, self.shape_liquid, self.l):
                    C_mat[i, j] = -1 if i in inh else 1
        
        return inh, C_mat
        
    def __getInputLayer(self):
        IL = []
        exh = list(set(range(self.n_neurons)) - set(self.inh))

        for nc in range(self.n_channels):
            aux = []
            for i in exh:
                umb = np.random.uniform(0, 1)
                if umb < self.prob_IL:
                    aux.append(i)
            IL.append(aux)
        
        mat = np.zeros((self.n_neurons, self.n_channels))
        for i in range(len(IL)):
            for j in IL[i]:
                mat[j][i] = 1
        
        return mat