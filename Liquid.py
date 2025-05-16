import numpy as np
from scipy.spatial import distance
from math import exp
import os

class Controller:
    def __init__(self, t_factor, W, gamma = 0.5, i_ext = 0.1, theta = 1):
        self.N = len(W)             # Cantidad de neuronas en la red
        self.gamma = gamma
        self.i_ext = i_ext
        self.theta = theta
        self.W = W                  # Matriz de pesos
        # self.V_0 = V_0              # Estado inicial
        self.t_factor = t_factor
        
       
    def reset(self, duracion):
        self.K = int(duracion + self.t_factor*duracion)                  # Tiempos de simulacion
        self.V = np.zeros((self.N, self.K+1))  
        self.Z = np.zeros_like(self.V, dtype=int)
        
    
    def cambio(self, i, k, entrada = None):
        acum = 0
        #reemplazar por if not entrada:
        if entrada is not None and len(entrada) != 0:
            acum += sum([self.W[i][j+self.N]*entrada[j] for j in range(len(entrada))])
        
        acum += sum([self.W[i][j]*self.Z[j][k-1] for j in range(self.N)])
        # print(type(acum))

        self.V[i][k] = float(self.gamma * self.V[i][k-1] * (1 - self.Z[i][k-1]) + acum + self.i_ext)
        self.Z[i][k] = 1 if self.V[i][k] >= self.theta else 0 
    
    
    # recibe función generadora tiempos que itera por cada uno de los tiempos del dato
    def simu(self, gen_tiempos, duracion):
        self.reset(duracion)
        t = 0
        for estado in gen_tiempos:
            for i in range(self.N):
                self.cambio(i, t, estado)
            t += 1
            if t%1000 == 0:
                print(t)

        for k in range(t, self.K):
            for i in range(self.N):
                self.cambio(i, k)
            if t%1000 == 0:
                print(t)

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

    return C * exp(- (D/l)**2)

# CLASS LIQUID

class Liquid:
    def __init__(self, shape_liquid: tuple, n_channels:int, t_factor:int, 
                 l:float=2.2, prob_IL:float=0.3, perc_inh:float = 0.2):

        # hiperparametros 
        self.l = l
        self.t_factor = t_factor
        self.prob_IL = prob_IL
        self.perc_inh = perc_inh
        
        self.shape_liquid = shape_liquid
        self.n_neurons = shape_liquid[0] * shape_liquid[1] * shape_liquid[2]
        
        self.inh, self.C_eq = self.__getConnMatrix()

        self.n_channels = n_channels
        
        self.inputLayer = self.__getInputLayer()

        W_random = np.random.uniform(0, 1, (self.n_neurons, self.n_neurons+self.n_channels))
        self.C_eq = np.hstack([self.C_eq, self.inputLayer])
        
        self.WC = W_random * self.C_eq

        self.cpg = Controller(self.t_factor, self.WC)

    # recibe función generadora inputs que itera por todo el conjunto de datos 
    # y para cada uno regresa una función generadora de los tiempos del dato
    
    def simulacion(self, gen_inputs):
        self.sim = []

        # Crear carpeta 'simulacion' si no existe
        os.makedirs("Simulacion", exist_ok=True)

        for idx, (gen_dato, duracion) in enumerate(gen_inputs):
            matriz = self.cpg.simu(gen_dato, duracion)
            self.sim.append(matriz)
            nombre_archivo = os.path.join("Simulacion", f"sim_{idx}.txt")
            np.savetxt(nombre_archivo, matriz, fmt="%.6f")  # Ajusta el formato si es necesario

        return self.sim

    '''def simulacion(self, gen_inputs):
        self.sim = []
        for gen_dato, duracion in gen_inputs:
            self.sim.append(self.cpg.simu(gen_dato, duracion))
        return self.sim
    '''    
    def photo(self, factor:float):
        n_frames = factor*self.t_factor
        
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