import eigenSolverNN as nn
import eigenSolverFE as fe
import tensorflow as tf
from tensorflow.experimental.numpy import logspace
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

learning_rate_range = np.array([0.0175])
epochs_range = np.array([10000])


grid_search_error = np.zeros([
    learning_rate_range.shape[0],
    epochs_range.shape[0]
])

#Array and initial conditions
Q = tf.random.normal([6,6], dtype = 'float64')
A = 0.5*(Q+tf.transpose(Q))
starting_point = tf.random.normal([6], dtype = 'float64')
def run():
    #Forward euler solution to test against
    forward_euler_solution = fe.forward_euler_sol(A, starting_point)[1][:, -1]

    for i, learning_rate in enumerate(learning_rate_range):
        for j, epochs in enumerate(epochs_range):
            neural_net = nn.eigSolverNN(A, starting_point)
            neural_net.optimizer.lr.assign(learning_rate)
            neural_net.train_model(int(epochs), 10)
            error = tf.reduce_mean(tf.square(
                forward_euler_solution - neural_net.compute_eig()[1].numpy()
            ))

            grid_search_error[i, j] = error

    with open('grid_search.pickle', 'wb') as file:
        pickle.dump(grid_search_error, file)

    return grid_search_error

def load():
    with open('grid_search.pickle', 'rb') as file:
        grid_search_error = pickle.load(file, encoding='bytes')
    return grid_search_error

if __name__ =='__main__':

    grid_search_error = run()

    for i, lr in enumerate(learning_rate_range):
        for j, ep in enumerate(epochs_range):
            print(lr, ep, grid_search_error[i, j])

    print(np.amin(grid_search_error))

    plt.imshow(grid_search_error)
    plt.show()
