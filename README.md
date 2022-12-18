
<h1 align="center">
  <br>
  FYS-STK4155 Project 3
  <br>
</h1>

<h4 align="center">Solving the diffusion equation, and diagonalizing a symmetrical matrix, with numerical methods and neural networks.</h4>

![](https://github.com/giammy00/FYS-STK4155-Project-3/blob/main/1M_animation.gif)
## Description

In this project we solve the diffusion eqaution:

$$ \frac{\partial^2 u}{\partial x^2} = \frac{\partial u}{\partial t} $$

by using the finite-difference method and with a neural network. We also look at the ordinary differential equation for finding eigenvectors of a symmetrical matrix:

$$     \frac{dx}{dt} = -x(t) + f( x (t))  $$

where 

$$     f ( x) =  \left[ x^\intercal  x A + (1- x^\intercal A x)I\right] x $$

and try to solve it with forward euler and with a neural network.

## Reproducing the results

+ To reproduce our training of the network to solve the diffusion equation, simply run the script `NNSolver.py` in a terminal. `Notebook_00.ipynb` contains the numerical solution and `Notebook_04.ipynb` contains relevant plots.
+ Diagonalization of a symmetric matrix is shown in notebooks 01, 02, 03 in the notebook folder.

## Files explained
+ notebooks
  + Notebook_00.ipynb - Notebook going through how we approximated the solution to the diffusion equation witht he finite-difference method
  + Notebook_01.ipynb - Notebook going through the process of diagonalizing a symmetric matrix with forward euler. 
  + Notebook_02.ipynb - Notebook where we train a NN to find an eigenvector of a symmetric matrix and investigate about why we do not find what we expect
  + Notebook_03.ipynb - Notebook going through our attempts to find all eigenvectors of a symmetric matrix
* Report.pdf - The report of our project.
* 1M_animation.gif - Animation of our approximate solution to the diffsuion equation using a neural net
* plotter.py - Different plotting functions
* FDSolver.py - Approximating the diffusion equation with finite-difference scheme
* eigenSolverNN.py - Contains a class for diagonalizing a symmetric matrix with a neural network
* eigenSolverFE.py - Diagonalizing a symmetric matrix with forward euler.
* Eigenvects.py - Functions for creating orthagonal vectors and using a trained neural net to try and find all eigenvectors of a symmetric matrix.
* smallweights.py - A test of using small initial weights for eigenSolverNN.py
* gridsearch.py - An attempt to find suitable values for neural net parameters.

## Authors
[Gianmarco Puleo](https://github.com/giammy00) <br>
[David Svejda](https://github.com/DavidSvejda2507)<br>
[Henrik Breitenstein](https://github.com/henrikbreitenstein)<br>




