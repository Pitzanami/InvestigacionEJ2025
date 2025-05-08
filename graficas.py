import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def graph(Z):
    K = list(range(len(Z[0])))
    n = len(Z)
    
    figs, axs = plt.subplots(figsize=(8, 6))
    colors = plt.cm.turbo(np.linspace(0, 1, n))
    
    for i, row in enumerate(Z):
        active_times = np.where(row == 1)[0]
        
        for t in active_times:
            axs.broken_barh([(t, 1)], (i - 0.4, 0.8), facecolors=colors[i])
            
    axs.set_xlabel("Tiempo")
    axs.set_ylabel("Objeto")
    axs.set_xlim(right = len(K))
    # axs.set_xticks(range(len(K)))
    # axs.set_yticks(range(n))
    axs.set_title("Diagrama de actividad")
    axs.grid(True, linestyle="--", alpha=0.5)

def anigraph(Z, name = "animacion"):
    n, K = Z.shape
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.turbo(np.linspace(0, 1, n))

    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Objeto")
    ax.set_xlim(0, K)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(K))
    ax.set_yticks(range(n))
    ax.set_title("Diagrama de actividad animado")
    ax.grid(True, linestyle="--", alpha=0.5)

    bars = []
    
    for i in range(n):
        row_bars = []
        for t in range(K):
            rect = patches.Rectangle((t, i - 0.4), 1, 0.8, color=colors[i], alpha=0)
            ax.add_patch(rect)
            row_bars.append(rect)
        bars.append(row_bars)
        
    def update(frame):
        for i in range(n):
            if Z[i, frame] == 1:
                bars[i][frame].set_alpha(1)
        return [bar for row in bars for bar in row]

    ani = animation.FuncAnimation(fig, update, frames=K, interval=100, repeat=True)
    ani.save(f"gifs/{name}.gif", writer=PillowWriter(fps=2))
    
def indexToCoords(shape, index):
    _, Y, Z = shape
    
    x = index // (Y*Z)
    y = (index % (Y*Z))//Z
    z = index % Z
    
    return (x, y, z)
    
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