import numpy as np
from scipy.spatial import distance
import math
import random
from pathlib import Path
from pyNAVIS import Loaders, MainSettings, Functions

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

def getConnMatrix(n, shape, l, perc = 0.2):
    inh = random.sample(range(n), int(n*perc))
    
    C_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            
            umb = random.uniform(0, 1)
            if umb < connProb(i, j, inh, shape, l):
                C_mat[i, j] = -1 if i in inh else 1
    
    return inh, C_mat

def load_file(file_path: Path, settings: MainSettings):
    
    loaded_file  = Loaders.loadAEDAT(path=str(file_path), settings=settings)
    adapted_file = Functions.adapt_timestamps(loaded_file, settings)
    
    return loaded_file

def load_directory(data_folder: Path, settings: MainSettings, file_extension: str = ".aedat", verbose:bool=False):
        
    all_data = []
    file_list = sorted(data_folder.glob(f"*{file_extension}"))
    
    for file_path in file_list:
        
        data = load_file(file_path, settings)
        addresses = data.addresses
        timestamps = data.timestamps
        max_timestamps = np.max(timestamps) if timestamps.size > 0 else 0
        
        all_data.append({
            "file_name": file_path.name,
            "addresses": addresses,
            "timestamps": timestamps,
            "max_timestamps": max_timestamps
        })
        
    print(f"Se cargaron {len(all_data)} archivos exitosamente desde {data_folder}")
        
    return all_data

