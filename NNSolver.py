import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython import display

from scipy.integrate import odeint

# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
layers = tf.keras.layers
from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import reshape
# import tensorflow.python.ops.numpy_ops as ops
# print(dir(ops))
# np_config.enable_numpy_behaviour()


inputs = tf.keras.Input(shape=(2))
x = layers.Dense(50, activation = "sigmoid")(inputs)
x2 = layers.Dense(20, activation = "sigmoid")(x)
output = layers.Dense(1)(x2)
# output = layers.Dense(1)(x2)

ode_solver = tf.keras.Model(inputs, output, name="ode_solver")
# ODE parameters
# g = 10
# l = 10
# theta_0 = tf.Variable(1, dtype ='float32')


# Sampling parameters
T = 1
# sampling_stages = 10**4
sample_size = 100 # not used when using non stochastic methods

#Training parameters
batch_size = 4
epochs = 10
epoch_size = 1000

ts = np.linspace(0,T,10)
x = np.linspace(0,1,10)
xs, ts = np.meshgrid(x, ts)
# print(f"{xs.shape=}\,{ts.shape=}")
xts = np.stack([xs.flatten(), ts.flatten()], axis = -1)
# print(f"{xts.shape=}")

init_lr = 5e-3
decay_rate = (1/7)**0.0
weight_decay = 1e-5
def weight_decay_func(s):
    # return 10**(-5 - 1*(s/epochs))
    return 0

# optimizer = tf.keras.optimizers.Adam(init_lr)
# optimizer = tf.keras.optimizers.SGD(init_lr)
optimizer = tf.keras.optimizers.Adam(init_lr)

def sampler(size, ep):
    out = np.zeros((size,2))
    # out[:,0] = np.random.uniform(low=0., high=(ep/epochs)*T,  size=(size)).astype(np.float32)
    out[:,0] = np.random.uniform(low=0., high=T,  size=(size)).astype(np.float32)
    out[:,1] = np.random.uniform(low=0., high=1., size=(size)).astype(np.float32)
    out[:,1] = 0.5+(ep/epochs)*(out[:,1]-0.5)
    return out
    # return xts

# t_interior = sampler(sample_size, 1)
t_interior = xts
t_interior_tf = tf.Variable(t_interior, trainable=False)
print(ode_solver(t_interior_tf).shape)
"""
def loss(model):
    # compute function value and derivatives at current sampled points
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_interior_tf)
        NN = model(t_interior_tf)
        NN = tf.cast(NN, np.float64)
        t = t_interior_tf[:,0]
        x = t_interior_tf[:,1]
        u = (1-t)*tf.sin(x*np.pi) + t*x*(1-x)*NN
        du = tape.gradient(u,t_interior_tf)

    ddu = tape.gradient(du,t_interior_tf)
    theta_err = ddu[:,1] - du[:,0]

    L1 = tf.reduce_mean(tf.square(theta_err))

    return L1

"""
def loss(model):
    # compute function value and derivatives at current sampled points
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_interior_tf)
        theta = model(t_interior_tf)
        theta_t = tape.gradient(theta,t_interior_tf)

    theta_tt = tape.gradient(theta_t,t_interior_tf)

    # Test #####################
    # xt = t_interior_tf
    # theta = tf.exp(-xt[:,1]*np.pi**2)*tf.sin(xt[:,0]*np.pi)
    # theta_t = [(np.pi**2)*tf.exp(-xt[:,1]*np.pi**2)*tf.sin(xt[:,0]*np.pi), tf.exp(-xt[:,1]*np.pi**2)*tf.cos(xt[:,0]*np.pi)*np.pi]
    # theta_tt = [(np.pi**4)*tf.exp(-xt[:,1]*np.pi**2)*tf.sin(xt[:,0]*np.pi), -tf.exp(-xt[:,1]*np.pi**2)*tf.sin(xt[:,0]*np.pi)*(np.pi**2)]
    # theta_t = tf.stack(theta_t, axis=1)
    # theta_tt = tf.stack(theta_tt, axis=1)
    # Test #####################

    # print(f"{theta=}")
    # print(f"{t_interior_tf=}")
    # print(f"\n\n\n{theta_t=},\n{theta_tt=}\n\n\n")
    # print(f"{theta=}")
    # print(f"{theta_t=}")
    # print(f"{theta_tt=}")
    # print(f"{t_interior_tf[:,0]=}")
    # print(f"{t_interior_tf[:,1]=}")
    t = t_interior_tf[:,0, np.newaxis]
    x = t_interior_tf[:,1, np.newaxis]
    N = tf.cast(theta, np.float64)#[:,0]

    # e1 = tf.sin(np.pi*x)
    # e2 = (t-1)*np.pi**2*e1
    # e3 = -2*t*N
    # e4 = t*(2-4*x)*theta_t[:,1]
    # e5 = x*(1-x)*(t*(theta_tt[:,1]-theta_t[:,0])-N)

    # theta_err = e1+e2+e3+e4+e5

    # print(f"{theta_err=}")
    # print(f"{e1=}")
    # print(f"{e2=}")
    # print(f"{e3=}")
    # print(f"{e4=}")
    # print(f"{e5=}")


    uxx = (-(np.pi**2)*(1-t)*tf.sin(np.pi*x)) + t*((2*(1-2*x)*theta_t[:,1:2]) - (2*N) + x*(1-x)*theta_tt[:,1:2])
    ut = -tf.sin(np.pi*x) + x*(1-x)*(N + t*theta_t[:,0:1])

    theta_err = uxx - ut
    print(theta_err.shape)
    # print(theta_err)
    # print(dir(theta_err))
    # # print(theta_err.numpy()s)

    L1 = tf.reduce_mean(tf.square(theta_err))
    # print(f"{L1=}")
    return L1
#"""
# def train_step(model):
#   @tf.function
#   def inner_func():
#     with tf.GradientTape() as tape:
#       # Compute the loss value for this minibatch.
#       L_ode = loss(model)
#
#     # Use the gradient tape to automatically retrieve
#     # the gradients of the trainable variables with respect to the loss.
#     grads = tape.gradient(L_ode, model.trainable_variables)
#     #grads = [grad + weight_decay * weight for grad, weight in zip(grads, model.weights)]
#
#     # Run one step of gradient descent by updating
#     # the value of the variables to minimize the loss.
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return L_ode, grads
#   return inner_func


@tf.function
def train_step(model):
    with tf.GradientTape() as tape:
      # Compute the loss value for this minibatch.
      L_ode = loss(model)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(L_ode, model.trainable_variables)
    #grads = [grad + weight_decay * weight for grad, weight in zip(grads, model.weights)]

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return L_ode, grads


def train_model(model):
  try:
    losses = []
    #train_step_function = train_step(model)

    start = time.time()
    for s in range(1,epochs+1):
        step_size = init_lr*decay_rate**s
        optimizer.lr.assign(step_size)
        #weight_decay = weight_decay_func(s)
        for _ in tqdm(range(epoch_size)):
            mean_loss = []
            # t_interior = xts
            # t_interior_tf.assign(t_interior)

            for _ in range(batch_size):
                #t_interior = sampler(sample_size, s)
                #t_interior_tf.assign(t_interior)
                current_loss, grads = train_step(model)
                # print(f"{current_loss.numpy()=}")
                mean_loss.append(current_loss)

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

    return np.array(losses)
  except KeyboardInterrupt:
    print("Interupted, making plots anyway")
    return np.array(losses)


losses = train_model(ode_solver)

ode_solver.save("modelH.H5")

def model(xt):
    return (1-xt[:,1])*np.sin(xt[:,0]*np.pi) + xt[:,0]*(1-xt[:,0])*xt[:,1] * ode_solver(tf.Variable(xt))[:,0]

def sol(xt):
    return np.exp(-xt[:,1]*np.pi**2)*np.sin(xt[:,0]*np.pi)

ts = np.linspace(0,T,100)
x = np.linspace(0,1,100)
xs, ts = np.meshgrid(x, ts)
# print(f"{xs.shape=}\,{ts.shape=}")
xts = np.stack([xs.flatten(), ts.flatten()], axis = -1)
# print(f"{xts.shape=}")

# print(xts)
# print(f"{(model(xts))=}")
model = reshape(model(xts),(100,100))
sol = sol(xts).reshape((100,100))

fig, axs = plt.subplots(2,1,figsize = (16,8), gridspec_kw={'height_ratios': [3, 1.5]})

line1 = axs[0].plot(x, sol[0], 'b', label='T(t)')
# print(type(line1))
line2 = axs[0].plot(x, model[0],'r', label='NN approximation')

axs[0].legend(loc='upper right')


title = axs[0].set_title('training batches: 0, mean squared error: 0, maximum absolute error: 0')
axs[0].grid()

# axs[1].set_ylim(-8.1,2)
# axs[1].set_xlim(0,(epochs-1)*epoch_size)
axs[1].grid()

loss_line, = axs[1].plot(np.log10(losses), label = 'Training loss')

# mae_line, = axs[1].plot([],[], label = 'maximum absolute erorr')
# mse_line, = axs[1].plot([],[], label = 'mean squared error')


axs[1].legend()
plt.show()

# log_losses = np.log10(losses)
# mae = []
# mse = []
# for y in simulations:
#   error = (sol - y[:,0])
#   mse.append(np.log10(np.mean(np.square(error))))
#   mae.append(np.log10(np.max(np.abs(error))))

def AnimationFunction(frame):
    line1[0].set_data(x, sol[frame])
    line2[0].set_data(x, model[frame])
    # x = t_points
    # y = np.array(simulations[frame])
    # nn_approx.set_data((x,y))

    # error = (sol - y[:,0])

    # title.set_text(f'training batches: {frame}')
    # running_x = list(range(frame))
    # loss_line.set_data((running_x,log_losses[:frame]))
    # mse_line.set_data((running_x, mse[:frame]))
    # mae_line.set_data((running_x, mae[:frame]))

# AnimationFunction(len(sol)-1)

# plt.savefig("test.png")

anim_created = FuncAnimation(fig, AnimationFunction, frames=len(model), interval=50)

# plt.rcParams['animation.ffmpeg_path'] = "/usr/lib/x86_64-linux-gnu"

anim_created.save("animation.gif", writer = "pillow", fps = 30)

# video = anim_created.to_html5_video()
# html = display.HTML(video)
# display.display(html)

plt.close()
