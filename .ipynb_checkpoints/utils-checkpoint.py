import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def poolData(x, polyorder):
    """Builds a polynomial library of candidate functions."""

    # If the input is a 1D array, convert it to a 2D array for consistent processing.
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples, n_vars = x.shape
    ind = 0
    # Add a column for the constant term (1)
    theta = np.ones((n_samples, 1))
    ind += 1

    # Poly order 1
    for i in range(n_vars):
        theta = np.hstack([theta, x[:, i:i+1]])
        ind += 1
    
    # Poly order 2
    if polyorder >= 2:
        for i in range(n_vars):
            for j in range(i, n_vars):
                theta = np.hstack([theta, (x[:, i] * x[:, j]).reshape(-1, 1)])
                ind += 1

    # Poly order 3
    if polyorder >= 3:
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    theta = np.hstack([theta, (x[:, i] * x[:, j] * x[:, k]).reshape(-1, 1)])
                    ind += 1
    
    return theta


def plot_sindy(t_data, x_data, t_sindy, x_sindy):
    """
    Compares original data with SINDy model simulation for the Lorenz system.
    """
    fig = plt.figure(figsize=(8, 10))
    
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(t_data, x_data[:, 0], 'b', label='Data x', linewidth=1.2)
    ax1.plot(t_sindy, x_sindy[:, 0], 'k--', label='SINDy x', linewidth=1)
    ax1.set_ylabel('x')
    ax1.set_xticklabels([])
    ax1.legend()
    
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(t_data, x_data[:, 1], 'r', label='Data y', linewidth=1.2)
    ax2.plot(t_sindy, x_sindy[:, 1], 'k--', label='SINDy y', linewidth=1)
    ax2.set_ylabel('y')
    ax2.set_xticklabels([])
    ax2.legend()

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(t_data, x_data[:, 2], 'g', label='Data z', linewidth=1.2)
    ax3.plot(t_sindy, x_sindy[:, 2], 'k--', label='SINDy z', linewidth=1)
    ax3.set_ylabel('z')
    ax3.set_xlabel('time')
    ax3.legend()

    ax4 = fig.add_subplot(5, 1, (4, 5), projection='3d')
    ax4.plot(x_data[:, 0], x_data[:, 1], x_data[:, 2], 'b', linewidth=0.5, label='Data')
    ax4.plot(x_sindy[:, 0], x_sindy[:, 1], x_sindy[:, 2], 'k--', linewidth=0.5, label='SINDy')
    ax4.set_xlim(-40, 40)
    ax4.set_ylim(-40, 40)
    ax4.set_zlim(-10, 50)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    #ax4.view_init(elev=-140, azim=20)
    ax4.legend()
    plt.tight_layout()
    plt.show()

def plot_pendulum(t, x, title='Pendulum Time Series Data'):
    """Plots the time series of the inverted pendulum system."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(t, x[:, 0], 'b')
    axs[0].set_ylabel('x')
    axs[0].set_xticklabels([])
    axs[0].set_title(title)
    axs[1].plot(t, x[:, 1], 'r')
    axs[1].set_ylabel('dx/dt')
    axs[1].set_xlabel('time')
    plt.tight_layout()
    plt.show()

def plot_sindy_pendulum(t_data, x_data, t_sindy, x_sindy):
    """Compares original pendulum data with SINDy model simulation."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(t_data, x_data[:, 0], 'b', label='Data x', linewidth=1.2)
    axs[0].plot(t_sindy, x_sindy[:, 0], 'k--', label='SINDy x', linewidth=1)
    axs[0].set_ylabel('x')
    axs[0].set_xticklabels([])
    axs[0].legend()
    axs[1].plot(t_data, x_data[:, 1], 'r', label='Data dx/dt', linewidth=1.2)
    axs[1].plot(t_sindy, x_sindy[:, 1], 'k--', label='SINDy dx/dt', linewidth=1)
    axs[1].set_ylabel('dx/dt')
    axs[1].set_xlabel('time')
    axs[1].legend()
    plt.tight_layout()
    plt.show()