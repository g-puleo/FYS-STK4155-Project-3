import os
from tqdm import tqdm
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages
import tensorflow as tf
tf.keras.backend.set_floatx('float64') #set default precision to double

#DEFINITION OF DEFAULT ARGS TO THE eigSolver.__init__ function.

#DEFINE DEFAULT NEURAL NET MODEL
N_hidden = 100
inputs = tf.keras.Input(shape=(1), name="time")
l1 = tf.keras.layers.Dense(N_hidden, activation="sigmoid")
out = tf.keras.layers.Dense(6, activation='linear')
model = tf.keras.Sequential( [inputs, l1, out])

#define default time grid (DATASET)
#note well: final time T=1 is chosen after looking at Forward Euler evolution
t_grid = tf.linspace(0, 1, 100)[:,tf.newaxis]

class eigSolverNN():
	'''class for diagonalization of a symmetric matrix A, by training of a neural network.
	The NN is rained to solve the ODE defined by Yi et al. in
	https://www.researchgate.net/publication/222949356_Neural_networks_based_approach_for_computing_eigenvectors_and_eigenvalues_of_symmetric_matrix
	The solution of the ODE is proven to converge to an eigenvector of the matrix A.

	Attributes:
		model: 			Tensorflow model which is used to fit the solution to the ODE
		A: 				ymmetric matrix to diagonalize
		x0: 			initial condition of ODE
		optimizer: 		tf.keras.optimizers.Optimizer which does SGD
		t_grid:			the grid of time steps where solution is wanted
		n_dim: 			A.shape[0] (dimension of space)
		Id: 			Identity matrix

		The following attributes are availablle after calling the method train_model()

		loss:			sequence of losses computed after every epoch of training
		eigvecs: 		sequence of vectors, which should approach an eigenvector of the matrix A as we train more
		eigvals: 		(supposed) eigenvalues corresponding to the eigvecs .

	Methods:

		f:				function f defined in the paper
		x_tilde: 		evaluate trial solution at time step or series of time steps
		loss:			compute loss function at given batch
		gradient_step: 	perform one GD step on a given batch
		train_model: 	trains the model performing SGD
		compute_eigs: 	computes current estimate of eigenvalue and eigenvector
		'''

	def __init__(self, A, x0, model=model, optimizer=tf.keras.optimizers.Adam(), t_grid = t_grid):
		'''standard constructor of eigSolver
		Args:
			A: 			tf.Tensor() of shape (n,n). It's the matrix to be diagonalized. Needs to be symmetric for the result to be meaningful.
						Make sure to use 'float64' as dtype.
			x0: 		initial condition of the ode, as tf.Tensor of shape (n,1),
			model: 		instance of tf.Model()( neural network model). Defaults to an instance of tf.Sequential,
						with a hidden layer made of 100 neurons.
			optimizer: 	instance of any subclass of tf.keras.optimizers.Optimizer(). Defaults to Adam.
			t_grid: 	whole grid of time steps where the solution is wanted and where the net will be trained.
						Defaults to a sequence of 2000 time steps evenly spaced in the interval [0,1].
			'''
		self.model = model
		self.A = A
		#track dimension of space
		self.n_dim = A.shape[0]
		#make a corresponding identity matrix
		self.Id =  tf.eye(self.n_dim, dtype='float64')

		self.optimizer = optimizer
		self.t_grid = t_grid

		self.x0 = x0



	#define function  f which enters ode

	@tf.function
	def f(self, x):
		'''
		args:
			x: tensor of shape (npoints, n_dim), where npoints is the number of points in the grid, n_dim is
			the dimensionality of the space.

		returns:
			out: tensor of the same shape as the input x, corresponding to the action of the function
			f(y)= [y^T y A - (1-y^T A y) I ] y, where the column vector y in this expression corresponds to every
			row of the tensor x in input.
		'''
		xT = tf.transpose(x) #(n x npoints)
		xxT = tf.square(tf.norm(x, axis=1))#shape (npoints,)
		xAxT = tf.einsum('ij, jk, ik -> i', x, self.A, x) #keep only diagonal elements of tensor outer product using einsum
		mat1 = tf.tensordot(self.A, xxT, axes=0)#shape (n,n,npoints), this is a pile of npoints matrices
		mat2 = tf.tensordot(self.Id, 1-xAxT, axes=0)#another stack of npoints matrices
		mat_tot = mat1+mat2 #this has shape(n,n,npoints)
		out = tf.einsum("ijk,kj->ki", mat_tot, x)
		return out



	def x_tilde(self, t):
		'''defines ansatz for solution of our ODE:
		args:
			t: input of shape (npoints,1)
			model: neural net model built with keras
		returns:
			trial solution at all times contained in t, as tensor of shape (npoints, n_dim)
		'''
		starting = (1-t)*tf.transpose(self.x0)
		model_part = t*self.model(t)
		return starting + model_part

	@tf.function
	def loss(self, t_in):
		'''computes mean squared difference between rhs and lhs of ode, computed at every timestep define in t_in.
		args:
			t_in: tensor of shape (N,1), containing batch of time samples where cost is evaluated
		returns:
			loss: cost function at given sampled points'''


		# compute derivatives of the trial solution with respect to input t, at all points specified by t_in
		with tf.GradientTape() as tape:
			tape.watch(t_in)
			x = self.x_tilde(t_in)

		x_t = tape.gradient(x,t_in)

		#define cost as a "MSE"
		x_err = x_t + x - self.f(x)
		L1 = tf.reduce_mean(tf.square(x_err))


		return L1

	@tf.function
	def gradient_step(self, t_in):
		'''perform gradient descent step.
		 args:
		 t_in: batch of points used to compute the gradient of loss

		 returns:
		 loss: loss at current batch '''


		#compute gradient of loss at t_in
		with tf.GradientTape() as tape:
			#model's trainable variables should be watched automatically by the gradienttape

			#compute the loss function
			L = self.loss(t_in)


		gradient = tape.gradient(L, self.model.trainable_variables)

		#update network parameters (gradient step)
		self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

		return L


	def train_model(self, Nepochs, Nbatches, tolerance=1e-3):
		'''perform SGD using self.t_grid as dataset.
			args: Nepochs: number of epochs for training
				  Nbatches: number of batches
			returns: None'''

		self.info()

		print(f"\n\n Training model with SGD using {Nbatches} batches and {Nepochs} epochs.")
		try:
			self.losses = []
			self.eigvals = []
			self.eigvecs = []
			temp_grid = self.t_grid

			if Nbatches==1:

				for epoch in tqdm(range(Nepochs)):
					#shuffle dataset at every epoch
					temp_grid = tf.random.shuffle(temp_grid)
					curr_loss = self.gradient_step(t_current)

					if curr_loss < tolerance:
						break
			else:

				for epoch in tqdm(range(Nepochs)):
					#shuffle dataset at every epoch
					temp_grid = tf.random.shuffle(temp_grid)

					for ii in range(Nbatches):
						t_current = temp_grid[ii::Nbatches,:]

						#update self.model using gradient_step
						curr_loss = self.gradient_step(t_current)

					if curr_loss < tolerance:
						break
				#store loss
				self.losses.append(curr_loss)

				#store eigvalues and eigvectors
				eigval, eigvec = self.compute_eig()
				self.eigvals.append(eigval)
				self.eigvecs.append(eigvec)

			#convert to tf.Tensor format
			self.losses = tf.stack(self.losses)
			self.eigvals = tf.stack(self.eigvals)
			self.eigvecs = tf.stack(self.eigvecs)

			print(f"losses were: initial {self.losses[0]}, last: {self.losses[-1]}")
			return None

		except KeyboardInterrupt:

			print(f"losses were: initial {self.losses[0]}, last: {self.losses[epoch-1]}")
			return None

	def compute_eig( self ) :
		'''compute current prediction for the eigenvector and the corresponding eigvalue.
		This is done computing x_tilde at the latest time step in the training dataset.

		returns:
			eigval: sequence of values, which should approach an eigenvalue.
			eigvec: sequence of vectors, which should approach the corresponding eigenvector.'''

		#store current prediction for eigenvector

		eigvec = self.x_tilde(self.t_grid[-1])
		#and the corresponding eigenvalue. Note that eigvec.shape = (1,6) and not (6,1)
		#so that we need to transpose it on the RHS of A
		eigval = (eigvec @ self.A @ tf.transpose(eigvec) )/(tf.norm(eigvec)**2)

		return eigval, eigvec


	def info ( self ):
		'''prints out information about current model'''
		print( "------ Current settings of model: ------\n",
		f"Model:\n{self.model}",
		f"Layers:\n{self.model.layers}",
		f"Number of neurons in hidden layer: {self.model.layers[1].input_shape[1]}",
		f"Optimizer: {self.optimizer._name}",
		f"Learning rate: {self.optimizer.lr}",
		f"Time grid has shape {self.t_grid.shape} and goes from {self.t_grid[0]} to {self.t_grid[-1]}",
		f"Starting from initial condition: {self.x0}", sep="\n")

		return
