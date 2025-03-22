import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter

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
    axs.set_xticks(range(len(K)))
    axs.set_yticks(range(n))
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
    ani.save(f"{name}.gif", writer=PillowWriter(fps=2))