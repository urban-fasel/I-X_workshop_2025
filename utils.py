import numpy as np
import matplotlib.pyplot as plt

def plot_data(t, X, title):
    """
    Plots the time-series data for a dynamic system.
    
    Args:
        t (np.ndarray): The time array.
        X (np.ndarray): The data array with shape (timesteps, variables).
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, X[:, 0], label='Prey (x)')
    plt.plot(t, X[:, 1], label='Predator (y)')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sindy_simulation(t, X_orig, X_sim, title):
    """
    Compares the original data with a SINDy model simulation.
    
    Args:
        t (np.ndarray): The time array.
        X_orig (np.ndarray): The original data with shape (timesteps, variables).
        X_sim (np.ndarray): The simulated data with shape (timesteps, variables).
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, X_orig[:, 0], 'o', label='Original Prey Data')
    plt.plot(t, X_orig[:, 1], 'o', label='Original Predator Data')
    plt.plot(t, X_sim[:, 0], '-', label='SINDy Model Prey', linewidth=2)
    plt.plot(t, X_sim[:, 1], '-', label='SINDy Model Predator', linewidth=2)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()