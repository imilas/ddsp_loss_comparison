import itertools
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
def loss_grad_grids(imitator_instrument_params,granularity_sizes,grad_fn):
    param_names = list(imitator_instrument_params["params"].keys())
    param_ranges = dict([[p,(-1,1)] for p in param_names]) 
    granularities = dict([[p,g] for p,g in zip(param_names,granularity_sizes)]) 
    
    # Create list of values for each param
    param_values = {
        k: np.linspace(param_ranges[k][0], param_ranges[k][1], granularities[k])
        for k in param_ranges
    }
    
    grids = np.meshgrid(*param_values.values(), indexing='ij')              # shape: [len(a), len(b), len(c)]
    combined = np.stack(grids, axis=-1)                      # shape: [len(a), len(b), len(c), n]
    
    # Create output array with same shape excluding last axis
    grid_losses = np.empty(combined.shape[:-1])
    grad_losses = np.empty(combined.shape)
    # Iterate and apply function
    it = np.nditer(grid_losses, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        idx = it.multi_index
        value = combined[idx]  # e.g., [a_val, b_val]
        prg ={"params": dict([(p,g) for p,g  in zip(param_names,value)])}
        (it[0],_),g = grad_fn(prg)
        # print()
        grad_losses[idx] = list(g["params"].values())
        it.iternext()
    return grids, grid_losses,grad_losses

def loss_3d_plot(grids,grid_losses,grad_losses,param_names):
    X, Y = grids  # unpack the meshgrid
    Z = grid_losses
    
    # Plot 3D loss surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # Label axes using param names
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_zlabel("Loss")
    ax.set_title("Loss Surface")
    
    plt.tight_layout()
    plt.show()

def loss_2d_plot(grids, grid_losses, grad_losses, param_names,targets=None):
    X, Y = grids
    Z = grid_losses

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cm.viridis)
    plt.colorbar(c, ax=ax, label='Loss')
    
    ax.plot(*targets, 'ro', markersize=6, label='Target Params')

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title("2D Loss Heatmap")

    plt.tight_layout()
    return plt

