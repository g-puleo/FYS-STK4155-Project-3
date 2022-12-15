import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

figsize = (3.313, 3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_eigenvector_components_convergence(time, time_evolution_array):
    """
    Takes the time evolution of the point and plots the path for each component.

    args:
        time: array (or tensor) containing N timesteps
        time_evolution_array: np.ndarray (or tf.Tensor) of shape (dim, N), dim=dimension of space.
    """
    fig = plt.figure(figsize=figsize)
    plt.xlabel('t')
    plt.ylabel('Eigenvector Components')

    for component in time_evolution_array:
        plt.plot(time, component, lw=2, zorder=1)
        plt.grid(visible=True)

    return fig

def plot_mean_squared_error(epochs, epoch_evolution_array):
    """
    Takes the evolution of the mean squared error with respect to
    the number of epochs and plots it
    """
    fig = plt.figure(figsize=figsize)
    plt.xlabel('N_{epochs}')
    plt.ylabel('MSE')
    plt.plot(epochs, epoch_evolution_array, zorder=1, lw=2)
    plt.grid(visible=True)
    return fig

def plot_estimated_eigenvetors(epochs, final_eigenvect_evolution):

    fig = plt.figure(figsize=figsize)
    plt.xlabel('N_{epochs}')
    plt.ylabel('Eigenvector Components')

    for component in final_eigenvect_evolution:
        plt.plot(epoch, component, zorder=1, lw=2)
        plt.grid(visible=True)
    return fig
