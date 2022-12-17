# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotter
import os

# %%
model = tf.keras.models.load_model("Model_1M_steps.H5")

x = np.linspace(0, 1, 1000)
t = np.array([0.1, 0.3])
ts, xs = np.meshgrid(t, x)
xts = np.stack([xs.flatten(), ts.flatten()], axis = -1)

# %%
u = model(xts).numpy()
u = u.reshape((2,1000))

# %%
t = [0.1, 0.3]
t1_index = 0
t2_index = 1

# %%

fig1 = plotter.plot_fd_solutions(u, x, t, t1_index, t2_index)
# fig2 = plotter.plot_fd_solutions(dx100, x100, t, t1_index, t2_index)
cwd = os.getcwd()
fig1.savefig(cwd + '/Plots/Model_1M.png')
# fig2.savefig('Plots/dx100.pdf')

plt.show()

# %%



