import numpy as np
from pathlib import Path
# from scipy.spatial import distance
from math import exp, sqrt
from pyNAVIS import Loaders, MainSettings, Functions
from os import makedirs, path

# Numba imports
from numba import njit, prange

# CuPy import for GPU
try:
    import cupy as cp
    try:
        # prueba de cuBLAS
        _ = cp.zeros((1,)).dot(cp.zeros((1,)))
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False

# === File loading functions (unchanged) ===
def load_file(file_path: Path, settings: MainSettings):
    return Loaders.loadAEDAT(path=str(file_path), settings=settings)

def load_directory(data_folder: Path, settings: MainSettings, file_extension: str = ".aedat"):
    all_data = []
    for file_path in sorted(data_folder.glob(f"*{file_extension}")):
        data = load_file(file_path, settings)
        timestamps = data.timestamps
        all_data.append({
            "file_name": file_path.name,
            "addresses": data.addresses,
            "timestamps": timestamps,
            "max_timestamps": np.max(timestamps) if timestamps.size>0 else 0
        })
    print(f"Se cargaron {len(all_data)} archivos desde {data_folder}")
    return all_data

# === Numba-optimized connectivity matrix ===

@njit(parallel=True)
def compute_distance_matrix(coords, out):
    N = coords.shape[0]
    for i in prange(N):
        for j in range(N):
            dx = coords[i,0] - coords[j,0]
            dy = coords[i,1] - coords[j,1]
            dz = coords[i,2] - coords[j,2]
            out[i,j] = sqrt(dx*dx + dy*dy + dz*dz)

@njit(parallel=True)
def make_prob_matrix(N, inh_mask, dist_mat, l, out):
    for i in prange(N):
        for j in range(N):
            if inh_mask[i] and inh_mask[j]:
                C = 0.1
            elif inh_mask[i] and not inh_mask[j]:
                C = 0.4
            elif not inh_mask[i] and inh_mask[j]:
                C = 0.2
            else:
                C = 0.3
            D = dist_mat[i,j]
            out[i,j] = C * exp(- (D/l)**2)


def getConnMatrix_numba(n_neurons, perc_inh, shape_liquid, l):
    # inhibitory mask
    inh = np.zeros(n_neurons, dtype=np.bool_)
    inh_idx = np.random.choice(n_neurons, int(n_neurons*perc_inh), replace=False)
    inh[inh_idx] = True

    # build coordinates
    Y, Z = shape_liquid[1], shape_liquid[2]
    coords = np.zeros((n_neurons,3), dtype=np.int32)
    for i in range(n_neurons):
        coords[i,0] = i // (Y*Z)
        coords[i,1] = (i % (Y*Z)) // Z
        coords[i,2] = i % Z

    # precompute distances
    dist_mat = np.empty((n_neurons,n_neurons), dtype=np.float64)
    compute_distance_matrix(coords, dist_mat)

    # compute connection probabilities
    prob_mat = np.empty_like(dist_mat)
    make_prob_matrix(n_neurons, inh, dist_mat, l, prob_mat)

    # sample connections
    randm = np.random.random(prob_mat.shape)
    mask = randm < prob_mat

    # build connectivity matrix: -1 for inhibitory source neuron, +1 otherwise
    sign = np.where(inh, -1, 1).astype(np.int8)
    C_mat = mask.astype(np.int8) * sign[:, None]

    exh = np.where(~inh)[0]
    return inh, exh, C_mat

# === Numba-optimized simulation ===

@njit
def step_neuron(i, V_prev, Z_prev, Wrow, N, gamma, i_ext, theta):
    s = i_ext
    for j in range(N): s += Wrow[j] * Z_prev[j]
    V = gamma * V_prev * (1 - Z_prev[i]) + s
    return V, 1 if V >= theta else 0

@njit(parallel=True)
def run_sim_numba(K, N, Wconn, gamma, i_ext, theta, Vs, Zs):
    for k in range(1, K+1):
        Z_prev = Zs[:, k-1]
        for i in prange(N):  # <- esta parte SÍ se puede paralelizar
            V_prev = Vs[i, k-1]
            V, z = step_neuron(i, V_prev, Z_prev, Wconn[i], gamma, i_ext, theta)
            Vs[i, k] = V
            Zs[i, k] = z
    return Vs, Zs


# === CuPy (GPU) simulation ===
def run_sim_cupy(K, N, Wconn, gamma, i_ext, theta):
    W_gpu = cp.asarray(Wconn)
    Vs = cp.zeros((N, K+1), dtype=cp.float32)
    Zs = cp.zeros((N, K+1), dtype=cp.int8)
    for k in range(1, K+1):
        s = W_gpu[:,:N].dot(Zs[:, k-1]) + i_ext
        Vs[:, k] = gamma * Vs[:, k-1] * (1 - Zs[:, k-1]) + s
        Zs[:, k] = (Vs[:, k] >= theta).astype(cp.int8)
    return cp.asnumpy(Zs)

def getInputLayer(n_neurons, n_channels, exh, prob_IL):
    mat = np.zeros((n_neurons, n_channels), dtype=np.int8)
    for nc in range(n_channels):
        for i in exh:
            if np.random.rand() < prob_IL:
                mat[i, nc] = 1
    return mat

# === Save results ===
def save_simulation(Z, output_folder="Simulacion", filename="simulation.txt"):
    """
    Guarda la matriz de spikes Z en un archivo de texto.
    """
    makedirs(output_folder, exist_ok=True)
    filepath = path.join(output_folder, filename)
    np.savetxt(filepath, Z, fmt="%d")
    print(f"Resultados guardados en: {filepath}")

# === Main entry example ===
if __name__ == "__main__":
    settings = MainSettings(mono_stereo=0, address_size=2, ts_tick=0.2, bin_size=2000, num_channels=64)
    data = load_directory(Path("On_Off/off_aedats"), settings)

    # parameters
    shape_liquid = (5,5,5)
    n_channels = 64
    l = 2.2
    perc_inh = 0.2
    prob_IL = 0.3
    n_neurons = np.prod(shape_liquid)

    inh, exh, C_eq = getConnMatrix_numba(n_neurons, perc_inh, shape_liquid, l)

    # build input layer (vectorizado)
    # ... (igual al original, vectorizado si se desea)
    inputLayer = getInputLayer(n_neurons, n_channels, exh, prob_IL)

    # combine connectivity and input layer
    # W_random y C_eq se construyen como antes
    Wconn = np.random.rand(n_neurons, n_neurons + n_channels).astype(np.float32)
    Wconn *= np.hstack([C_eq, inputLayer])

    dur = int(data[0]['max_timestamps'] - data[0]['timestamps'][0])
    t_factor = 0.01
    K = int(dur + t_factor * dur)
    gamma = 0.5
    i_ext = 0.1
    theta = 1

    Vs = np.zeros((n_neurons, K+1), dtype=np.float32)
    Zs = np.zeros((n_neurons, K+1), dtype=np.int8)

    if GPU_AVAILABLE:
        try:
            print("Usando GPU para simulación...")
            Z_out = run_sim_cupy(K, n_neurons, Wconn, gamma, i_ext, theta)
        except Exception as e:
            print("Error en GPU (fallando a CPU):", e)
            Z_out = run_sim_numba(K, n_neurons, Wconn, gamma, i_ext, theta, Vs, Zs)[1]
    else:
        print("GPU no disponible, usando Numba CPU")
        Z_out = run_sim_numba(K, n_neurons, Wconn, gamma, i_ext, theta, Vs, Zs)[1]

    print("Simulación completa, output shape:", Z_out.shape)
    save_simulation(Z_out)
