import numpy as np

class CPG:
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
    
    