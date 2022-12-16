#simple solver class for the fwd difference solution of the heat equation
import numpy as np

class FDSolver():

	def check_bc(self):

		'''check that the boundary condition is satisfied, if not raise ValueError'''

		if np.abs(self.u[0])>1e-15 or np.abs(self.u[-1])>1e-15:
			raise ValueError("The boundary conditions are not met")

	def check_convergence(self):

		if self.alpha > 0.5:
			raise RuntimeWarning("dt/dx^2 is greater than 0.5. Algorithm will probably not converge.")

	def __init__(self, u0: np.array, dx, dt):

		'''Initialize solver. Args:
			u0: array containing initial condition (u0(x) at every x at time t=0)
			dx: size of space step (note that this must have been used to compute u0)
			dt: size of time step
		'''
		self.u = u0
		self.check_bc()
		self.alpha = dt/dx**2
		self.check_convergence()


	def evolve_fd(self):
		'''Evolve the function u at every location by one time step'''

		#compute difference between consecutive elements (approximately prop. to du/dx)
		diff1 = np.ediff1d(self.u)
		#do it again, to compute approximate second derivative
		diff2 = np.ediff1d(diff1)
		#update state of function u
		self.u[1:-1]+=self.alpha*diff2
