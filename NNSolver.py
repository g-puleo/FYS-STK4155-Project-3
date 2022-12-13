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

from tensorflow.python.ops.numpy_ops import reshape
# import tensorflow.python.ops.numpy_ops as ops
# print(dir(ops))
# np_config.enable_numpy_behaviour()


inputs = tf.keras.Input(shape=(2))
x = layers.Dense(256, activation = "sigmoid")(inputs)
output = layers.Dense(1)(x)

ode_solver = tf.keras.Model(inputs, output, name="ode_solver")

# ODE parameters
# g = 10
# l = 10
# theta_0 = tf.Variable(1, dtype ='float32')


# Sampling parameters
T = 3
# sampling_stages = 10**4
sample_size = 2**10

#Training parameters
batch_size = 2**4
epochs = 21
epoch_size = 100

def sampler(size):
    out = np.zeros((size,2))
    out[:,0] = np.random.uniform(low=0., high=T,  size=(size)).astype(np.float32)
    out[:,1] = np.random.uniform(low=0., high=1., size=(size)).astype(np.float32)
    return out

t_interior = sampler(sample_size) 
t_interior_tf = tf.Variable(t_interior, trainable=False)

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
    t = t_interior_tf[:,0]
    x = t_interior_tf[:,1]
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
    
    
    uxx = (-(np.pi**2)*(1-t)*tf.sin(np.pi*x)) + t*((2*(1-x)*theta_t[:,1]) - (2*N) + x*(1-x)*theta_tt[:,1])
    ut = -tf.sin(np.pi*x) + x*(1-x)*(N + t*theta_t[:,0])
    
    theta_err = uxx - ut
    
    # print(theta_err)
    # print(dir(theta_err))
    # # print(theta_err.numpy()s)
    
    L1 = tf.reduce_mean(tf.square(theta_err))
    # print(f"{L1=}")
    return L1

optimizer = tf.keras.optimizers.Adam(0.1)

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


def train_model(model):
  try:
    losses = []
    train_step_function = train_step(model)
    
    start = time.time()
    for s in range(1,epochs):
        step_size = 1e-1*0.7**s
        optimizer.lr.assign(step_size)     
        for _ in tqdm(range(epoch_size)):
            mean_loss = []
            t_interior = sampler(sample_size) 
            t_interior_tf.assign(t_interior)
            
            for _ in range(batch_size): 
                current_loss, grads = train_step_function()
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

def model(xt):
    return (1-xt[:,1])*np.sin(xt[:,0]*np.pi) + xt[:,0]*(1-xt[:,0])*xt[:,1] * ode_solver(tf.Variable(xt))[:,0]

def sol(xt):
    return np.exp(-xt[:,1]*np.pi**2)*np.sin(xt[:,0]*np.pi)

ts = np.linspace(0,1,100)
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

axs[1].set_ylim(-8.1,2)
axs[1].set_xlim(0,1000)
axs[1].grid()

loss_line, = axs[1].plot(np.log10(losses), label = 'Training loss')

# mae_line, = axs[1].plot([],[], label = 'maximum absolute erorr')
# mse_line, = axs[1].plot([],[], label = 'mean squared error')


axs[1].legend()
# plt.show()

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