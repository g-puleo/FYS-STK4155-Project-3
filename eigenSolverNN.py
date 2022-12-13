import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages
import tensorflow as tf
tf.keras.backend.set_floatx('float64') #set default precision to double





#DEFINE NEURAL NET MODEL
N_hidden = 100
inputs = tf.keras.Input(shape=(1), name="time")
l1 = tf.keras.layers.Dense(N_hidden, activation="sigmoid")
out = tf.keras.layers.Dense(6, activation='linear')

model = tf.keras.Sequential( [inputs, l1, out])

#define time grid and initial condition
#note well: final time T=1 is chosen after looking at Forward Euler evolution
t_in = tf.linspace(0,1, 2000)[:,tf.newaxis]
x_0 = tf.ones((6,1), dtype='float64')#just to test use ones()

#define matrix to diagonalize
Q = tf.random.normal((6,6), dtype='float64', seed=121222)
A = 0.5*(Q+tf.transpose(Q))
eigval, eigvec = tf.linalg.eigh(A)
print(f"eigvalues:{eigval}")
print(f"eigvectors: {eigvec}")
#and also the identity
Id6 = tf.eye(6, dtype='float64')

#define training parameters
optimizer = tf.keras.optimizers.Adam()
learning_rate = 1e-3
optimizer.lr.assign(learning_rate)
Nepochs = 10000

#define function  f which enters ode
def f(x):
	'''args:
	x: tensor of shape (npoints, n), where npoints is the number of points in the grid.
	'''
	n = A.shape[0]
	xT = tf.transpose(x) #(n x npoints)
	xxT = tf.square(tf.norm(x, axis=1))#shape (npoints,)
	xAxT = tf.einsum('ij, jk, ik -> i', x, A, x)
	mat1 = tf.tensordot(A, xxT, axes=0)#shape (n,n,npoints), this is a pile of npoints matrices
	mat2 = tf.tensordot(Id6, 1-xAxT, axes=0)#another stack of npoints matrices
	mat_tot = mat1+mat2 #shape(n,n,npoints)
	out = tf.einsum("ijk,kj->ki", mat_tot, x)
	return out



def x_tilde(t, model):
	'''defines ansatz for solution of our ODE:
	args:
		t: input of shape (n,1)
		model: neural net model built with keras
	'''
	starting = (1-t)*tf.transpose(x_0)
	model_part = t*model(t)
	return starting + model_part


def loss(model):
	'''computes mean squared difference between rhs and lhs of ode, computed at every timestep.
	args:
		model (tf.Model neural network)'''

	# compute function value and derivatives at current sampled points
	with tf.GradientTape() as tape:
		tape.watch(t_in)
		x = x_tilde(t_in, model)
	
	x_t = tape.gradient(x,t_in)

	x_err = x_t + x - f(x)
	L1 = tf.reduce_mean(tf.square(x_err))

	return L1


def gradient_step():
	with tf.GradientTape() as tape:
		#model's trainable variables should be watched automatically
		
		#compute the loss function 
		L = loss(model)


	gradient = tape.gradient(L, model.trainable_variables)
	optimizer.apply_gradients(zip(gradient, model.trainable_variables))
	#print("###gradient is:\n")
	#print(gradient)
	#print('\n\n\n ###after update trainable variables of model:')
	#print(model.trainable_variables)
	return L




def train_model():

	try:
		losses = []
		for epoch in tqdm(range(Nepochs)):
			curr_loss = gradient_step() 
			losses.append(curr_loss.numpy())
			# print(f"\n\n\n ####ITERATION NR. {ii}\n\n")
			# print(f"\ncurr_loss={curr_loss}\n")

		print(f"losses were: initial {losses[0]}, last: {losses[-1]}")
		return losses
	except KeyboardInterrupt:

		print(f"losses were: initial {losses[0]}, last: {losses[-1]}")
		return losses



losses = train_model()

print(tf.linalg.eigh(A))
print(x_tilde(1*tf.ones((1,1), dtype='float64'), model) )



final_evo = x_tilde(t_in, model).numpy()


for ii in range(6):
	plt.plot(t_in, final_evo[:,ii])

plt.title("elements of eigenvector")
plt.figure()

plt.plot(np.arange(len(losses)), losses)
plt.title("loss")
plt.show()