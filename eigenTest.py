import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython import display

from scipy.integrate import odeint

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

inputs = tf.keras.Input(shape=(1), name="time")

x = layers.Dense(32)(inputs)
x = layers.Dense(32)(x)
x = layers.Dense(8,  activation = tf.sin)(x)

x2 = layers.Dense(32)(inputs)
x2 = layers.Dense(32)(x2)
x2 = layers.Dense(8, activation = tf.exp)(x2)
output = layers.multiply([x,x2])
output = layers.Dense(6)(output)

ode_solver = tf.keras.Model(inputs, output, name="ode_solver")
ode_solver.summary()

# ODE parameters
x_0 = tf.Variable([1, 1, 1, 1, 1, 1], dtype ='float32')


# Sampling parameters
T = 30
sampling_stages = 10**4
sample_size = 2**10

#Training parameters
batch_size = 2**7
epochs = 11
epoch_size = 100

def sampler(size):
  return np.random.uniform(low=0.0, high=T, size=(size,1)).astype(np.float32)

t_interior = sampler(sample_size)
t_interior_tf = tf.Variable(t_interior, trainable=False)

A = np.random.randn(6, 6)
I = np.eye(6)

def f(x):
    n = np.shape(x)[0]
    spaced_norm = (np.linalg.norm(x, axis=1)**2)[:, np.newaxis, np.newaxis]
    xTA = np.tensordot(x, A, axes=1)
    xTAx = np.zeros(n)
    for i in range(n):
        xTAx[i] = np.dot(xTA[i], x[i])
    one = np.ones(n)
    temp1 = spaced_norm*A#x.TxA
    temp2 = (one-xTAx)[:, np.newaxis, np.newaxis]*I#(1-x.TAx)I
    temp3 = temp1 + temp2 #x.TxA + (1-x.TAx)I
    ret = np.zeros((n, 6))
    for i in range(n):
        ret[i, :] = temp3[i]@x[i]
    #[x.TxA + (1-x.TAx)I]x
    return ret


def loss(model):
    # compute function value and derivatives at current sampled points
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_interior_tf)
        x = model(t_interior_tf)
        x_t = tape.gradient(x,t_interior_tf)
    x_err = x_t + x - f(x)

    L1 = tf.reduce_mean(tf.square(x_err))

    zeros = tf.zeros(shape = (1,6), dtype=tf.dtypes.float32)
    zed_x = model(zeros)
    L2 = tf.reduce_mean(tf.square(zed_x - x_0))

    return (L1,L2)

optimizer = tf.keras.optimizers.Adam()

def train_step(model):
  @tf.function
  def inner_func():
    with tf.GradientTape() as tape:
      # Compute the loss value for this minibatch.
      L_ode = tf.reduce_sum(loss(model))

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(L_ode, model.trainable_variables)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L_ode, grads
  return inner_func

t_points = np.linspace(0,10,1000)

t_points_tf = tf.cast(tf.Variable(t_points, trainable=False),dtype=tf.float32)

def train_model(model):
  try:
    simulations = []
    losses = []
    t_points = np.linspace(0,10,1000)
    t_points_tf = tf.cast(tf.Variable(t_points, trainable=False),dtype=tf.float32)
    train_step_function = train_step(model)

    start = time.time()
    step_size = 1e-3
    optimizer.lr.assign(step_size)
    for s in range(1,epochs):
      for _ in tqdm(range(epoch_size)):
        mean_loss = []
        t_interior = sampler(sample_size)
        t_interior_tf.assign(t_interior)

        for _ in range(batch_size):
          current_loss, grads = train_step_function()
          mean_loss.append(current_loss)

        simulations.append(model(t_points_tf))
        mean_loss = np.array(mean_loss).mean()
        losses.append(mean_loss)

      end = time.time()
      print(
      f"""Training loss of value func at step {s*100}:
      loss: {mean_loss}.
      ######################
      """)

      #print("Step: %s" % s)
      #print("Seen so far: %s samples" % ((s + 1) * s_x_interior))
      print("Time: %.2f" % (end - start))
      print("-----------")
      print('Learning rate:  %.2e' % optimizer.learning_rate.numpy())

      start = time.time()

    return np.array(simulations), np.array(losses)
  except KeyboardInterrupt:
    return np.array(simulations), np.array(losses)

simulations, losses = train_model(ode_solver)
