# Code adapted from https://colab.research.google.com/gist/fjpAntunes/35aa78d4b48c68177f7340ccaf99720a/post-tensorflow-ode.ipynb
# Which was found through https://medium.com/@fjpantunes2/tensorflow-and-differential-equations-a-simple-example-77d88d98ea3e


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

layers = tf.keras.layers
from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import reshape


inputs = tf.keras.Input(shape=(2))
x = layers.Dense(50, activation = "sigmoid")(inputs)
x2 = layers.Dense(20, activation = "sigmoid")(x)
output = layers.Dense(1)(x2)

ode_solver = tf.keras.Model(inputs, output, name="ode_solver")

# Sampling parameters
T = 1 #final time

#Training parameters
epochs = 100
epoch_size = 100
batch_size = 100

ts = np.linspace(0,T,20)
x = np.linspace(0,1,20)
xs, ts = np.meshgrid(x, ts)
xts = np.stack([xs.flatten(), ts.flatten()], axis = -1)
RANDOM_RESAMPLING = True

init_lr = 5e-3
decay_rate = 1 #learning rate decay
weight_decay = 0 #ridge regrassion parameter
def weight_decay_func(s):
    return 0

# optimizer = tf.keras.optimizers.Adam(init_lr)
# optimizer = tf.keras.optimizers.SGD(init_lr)
optimizer = tf.keras.optimizers.Adam(init_lr)

def sampler(ep):
    out = np.zeros_like(xts)
    out[:,0] = np.random.uniform(low=0., high=T,  size=(out.shape[0])).astype(np.float32)
    out[:,1] = np.random.uniform(low=0., high=1., size=(out.shape[0])).astype(np.float32)
    out[:,1] = 0.5+(ep/epochs)*(out[:,1]-0.5)
    return out

t_interior = xts
t_interior_tf = tf.Variable(t_interior, trainable=False)

def loss(model):
    # compute function value and derivatives at current sampled points
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_interior_tf)
        theta = model(t_interior_tf)
        theta_t = tape.gradient(theta,t_interior_tf)

    theta_tt = tape.gradient(theta_t,t_interior_tf)

    t = t_interior_tf[:,1, np.newaxis]
    x = t_interior_tf[:,0, np.newaxis]
    N = tf.cast(theta, np.float64)

    uxx = (-(np.pi**2)*(1-t)*tf.sin(np.pi*x)) + t*((2*(1-2*x)*theta_t[:,0:1]) - (2*N) + x*(1-x)*theta_tt[:,0:1])
    ut = -tf.sin(np.pi*x) + x*(1-x)*(N + t*theta_t[:,1:2])

    theta_err = uxx - ut

    L1 = tf.reduce_mean(tf.square(theta_err))
    return L1


@tf.function
def train_step(model, weight_decay):
    with tf.GradientTape() as tape:
      # Compute the loss value
      L_ode = loss(model)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(L_ode, model.trainable_variables)
    if (weight_decay > 0):
        grads = [grad + weight_decay * weight for grad, weight in zip(grads, model.weights)]

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L_ode, grads


def train_model(model):
  try:
    losses = []

    start = time.time()
    for s in range(1,epochs+1):
        step_size = init_lr*decay_rate**s
        optimizer.lr.assign(step_size)
        weight_decay = weight_decay_func(s)
        for _ in tqdm(range(epoch_size)):
            mean_loss = []

            for _ in range(batch_size):
                if (RANDOM_RESAMPLING):
                    t_interior = sampler(s)
                    t_interior_tf.assign(t_interior)
                current_loss, grads = train_step(model, weight_decay)
                mean_loss.append(current_loss)

            mean_loss = np.array(mean_loss).mean()
            losses.append(mean_loss)

        end = time.time()
        print(
        f"""Training loss of value func at step {s*epoch_size}:
        loss: {mean_loss}.
        ######################
        """)

        print("Time: %.2f" % (end - start))
        print("-----------")
        print('Learning rate:  %.2e' % optimizer.learning_rate.numpy())

        start = time.time()

    return np.array(losses)
  except KeyboardInterrupt:
    print("Interupted, making plots anyway")
    return np.array(losses)


losses = train_model(ode_solver)

ode_solver.save("modelH.H5")
np.save("Losses", losses)

def model(xt):
    return (1-xt[:,1])*np.sin(xt[:,0]*np.pi) + xt[:,0]*(1-xt[:,0])*xt[:,1] * ode_solver(tf.Variable(xt))[:,0]

def sol(xt): # Analytical solution
    return np.exp(-xt[:,1]*np.pi**2)*np.sin(xt[:,0]*np.pi)

# Redo the grid with more points for plotting
ts = np.linspace(0,T,100)
x = np.linspace(0,1,100)
xs, ts = np.meshgrid(x, ts)
xts = np.stack([xs.flatten(), ts.flatten()], axis = -1)

model = reshape(model(xts),(100,100))
sol = sol(xts).reshape((100,100))
np.save("Model_output", model)

# Make the plot
fig, axs = plt.subplots(2,1,figsize = (16,8), gridspec_kw={'height_ratios': [3, 1.5]})

line1 = axs[0].plot(x, sol[0], 'b', label='T(t)')
line2 = axs[0].plot(x, model[0],'r', label='NN approximation')

axs[0].legend(loc='upper right')
title = axs[0].set_title(f'training batches: {epochs*epoch_size*batch_size}, loss: {losses[-1]}')
axs[0].grid()

loss_line, = axs[1].plot(np.log10(losses), label = 'Training loss')

axs[1].grid()
axs[1].legend()
plt.show()

# Make the animation
def AnimationFunction(frame):
    line1[0].set_data(x, sol[frame])
    line2[0].set_data(x, model[frame])

anim_created = FuncAnimation(fig, AnimationFunction, frames=len(model), interval=50)

anim_created.save("animation.gif", writer = "pillow", fps = 30)

# video = anim_created.to_html5_video()
# html = display.HTML(video)
# display.display(html)

plt.close()
