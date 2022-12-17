
<h1 align="center">
  <br>
  FYS-STK4155 Project 3
  <br>
</h1>

<h4 align="center">Solving the diffusion equation, and diagonalizing a symmetrical matrix, with numerical methods and neural networks.</h4>

$$ \frac{\partial^2 u}{\partial x^2} = \frac{\partial u}{\paritial t} $$

## Files explained
* 1M_animation.gif - Animation of our approximate solution to the diffsuion equation using a neural net
* plotter.py - Different plotting functions
* FDSolver.py - Approximating the diffusion equation with finite-difference scheme
* eigenSolverNN.py - Contains a class for diagonalizing a symmetric matrix with a neural network
* eigenSolverFE.py - Diagonalizing a symmetric matrix with forward euler.
* Eigenvects.py - Functions for creating orthagonal vectors and using a trained neural net to try and find all eigenvectors of a symmetric matrix.
* smallweights.py - A test of using small initial weights for eigenSolverNN.py
* gridsearch.py - An attempt to find suitable values for neural net parameters.
