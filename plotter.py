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
    plt.xlabel('time [-]')
    plt.ylabel('Eigenvector Components')

    #transpose to make it of shape (dim, N), for loop happens along axis 0.
    for component in time_evolution_array:
        plt.plot(time, component, lw=2, zorder=1)
        x_cordinate = time[-1]
        y_cordinate = component[-1]
        y_lims = plt.gca().get_ylim()
        plt.vlines(time[-1], y_lims[0], y_lims[1], linestyles='dotted', colors='gray', zorder=2)
        plt.scatter(x_cordinate, y_cordinate, s=10, color='red', zorder=4)
        plt.text(x_cordinate*1.01, y_cordinate*1.01, f'{y_cordinate:.2f}')

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
    plt.scatter(epochs[-1], epoch_evolution_array[-1], color='red', zorder=2)
    plt.text(epochs[-1]*1.01, epoch_evolution_array[-1]*1.01, f'{epoch_evolution_array[-1]}')

    return fig

def plot_estimated_eigenvetors(epochs, final_eigenvect_evolution):

    fig = plt.figure(figsize=figsize)
    plt.xlabel('N_{epochs}')
    plt.ylabel('Eigenvector Components')

    for component in final_eigenvect_evolution:
        plt.plot(epoch, component, zorder=1, lw=2)

    return fig
