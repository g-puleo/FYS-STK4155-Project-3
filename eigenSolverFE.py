#use forward euler to solve ODE to get eigvalues of A
import numpy as np

def f(x, A):
	n = A.shape[0]
	operator = (x.T @ x) * A + (1- x.T@A@x )*np.eye(n)
	return operator @ x


def forward_euler_sol(A, x0, Tmax=1, Nsteps=2000):
	'''compute solution of ODE using forward euler.
	args:
		A: 	np.ndarray (matrix of shape (n,n) to be diagonalized, must be numpy)
		x0:	np.array containing the initial condition
		Tmax: maximum timestep, defaults to 1
		Nsteps: number of timesteps, defaults to 2000.

	returns:
		t:	np.array of timesteps
		x: 	np.ndarray of shape (n,Nsteps)) containing the numerical solution
	'''
	n=A.shape[0]
	x = np.empty((n,Nsteps))
	#np.random.shuffle(x)
	t, dt = np.linspace(0,Tmax, Nsteps, retstep=True)
	x[:,0] = x0
	for i in range(Nsteps-1):
		xlast = x[:,i:i+1]
		x[:,i+1:i+2] = xlast + dt*(f(xlast, A)-xlast)



	return t, x
