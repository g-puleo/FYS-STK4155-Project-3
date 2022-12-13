#use forward euler to solve ODE to get eigvalues of A

import numpy as np
import matplotlib.pyplot as plt


Q = np.random.randn(6,6)
A = 0.5*(Q+Q.T)

A = np.array([0.7663,0.4283,-0.3237,-0.4298,-0.1438,0.4283,0.2862,0.0118,-0.2802,0.123,-0.3237,0.0118,-0.9093,-0.4384,0.7684,-0.4298,-0.2802,-0.4384,-0.0386,-0.1315,-0.1438,0.123,0.7684,-0.1315,-0.448])
A=A.reshape(5,5)
n=A.shape[0]

eigval, eigvec = np.linalg.eigh(A)
print(f"eigvalues:{eigval}")
print(f"eigvectors: {eigvec}")
def f(x):
	operator = (x.T @ x) * A + (1- x.T@A@x )*np.eye(n)
	return operator @ x

#initial cond

Nsteps = 100
Tfinal = 3
x = np.empty((n,Nsteps))
x[:,0] = np.array([0.68, 0.5, 0.27,0.58, 0.82])
#np.random.shuffle(x)
t = np.zeros(Nsteps)
dt = Tfinal/Nsteps

for i in range(Nsteps-1):
	xlast = x[:,i:i+1]
	x[:,i+1:i+2] = xlast + dt*(f(xlast)-xlast)
	t[i+1] = t[i]+dt



for jj in range(n):
	plt.plot(t, x[jj,:], label=f"{jj}")

plt.legend()
plt.show()