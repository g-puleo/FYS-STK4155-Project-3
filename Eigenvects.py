import eigenSolverNN as esnn
import numpy as np
import tensorflow as tf
import warnings

def normalized_proj(vector1, vector2):
    '''computes the projection of vector1 along the direction specified by vector2.
       With dirac's notation this is equivalent to ( <v2|v1>/<v2|v2> ) |v2>.
    '''
    normalization = np.dot(vector1, vector2)/np.dot(vector2, vector2)
    return normalization*vector2

def create_orthogonal(vectors_array):
    """
    Creates a random vector and substracts all components parallel to
    the vectors in vectors_array. These vectors belong to R^n .

    returns:
    orthogonal_vector: vector such that the dot product
         orthogonal_vector * vectors_array[ii] = 0 for all ii.
        IMPORTANT NOTE: this is true under the assumption that vectors_array is an orthogonal set.
        Note that this will be the 0 vector if there are more than n vectors in vectors_array .
    """
    n = len(vectors_array[0]) #size of vectors
    orthogonal_vector = np.random.randn(n)

    # Gram-Schmidt algorithm
    for vector in vectors_array:
        orthogonal_vector -= normalized_proj(orthogonal_vector, vector)

    return orthogonal_vector/np.linalg.norm(orthogonal_vector)

def check_eig(eigval, eigvec, A, tolerance=0.1):
    '''check that eigvec is an eigenvector of A with eigenvalue eigval.
    All args must be given as tf.Tensor() instances, and are converted to numpy here.
    '''

    eigval = eigval.numpy()
    eigvec = eigvec.numpy()
    A = A.numpy()

    #lambdavec should have components all equal to eigval
    lambdavec = (A@eigvec.T)/eigvec.T
    #dividing by eigval should give a vector of ones
    ones_vec = lambdavec/eigval
    print("For a perfect eigenvector, the following should be all 1.000")
    print(ones_vec)
    #check that this vector is close enough to np.ones()
    A = np.allclose(ones_vec, np.ones(ones_vec.shape), rtol=tolerance)
    if not A:
        warnings.warn(f"eigenvector ({eigvec}) might not be an eigenvector of A.")
    return A

def findEigenvectors(A):
    '''
    Find all eigenvectors of matrix A, using neural net.

    '''
    n = A.shape[0]

    eigenvectors = np.zeros((n,n))
    eigenvalues = np.zeros(n)

    starting_point = tf.random.normal([n], dtype='float64')

    #initialize instance of solver
    Nepochs = 50000
    Nbatches = 4

    for i in range(n):
        print(f"### FINDING EIGENVECTOR NR.{i} ###\n")
        solver = esnn.eigSolverNN(A, starting_point)
        solver.train_model(Nepochs, Nbatches)
        eigenvalue, eigvector = solver.compute_eig()

        #SHOULD CHECK THAT WHAT WE HAVE GOTTEN SO FAR *IS* AN EIGENVECTOR.
        #OTHERWISE, TRAINING AGAIN WOULD NOT MAKE SENSE
        check_eig(eigenvalue, eigvector, A)


        eigenvectors[i] = eigvector
        eigenvalues[i] = eigenvalue
        starting_point = create_orthogonal(eigenvectors[:i+1, :])

    return eigenvectors, eigenvalues


def Search_Until_Find(A, Nattempts=10, Nepochs=50000, Nbatches=1):
    '''
    Try to find all eigenvectors of A by training many neural networks to solve the ODE.
    Args:
        A:          matrix to diagonalize
        Nattempts:  number of models which you want to train. 
        Nepochs:    number of epochs for training each model
        Nbatches:   number of batches for SGD . Defaults to 1 (non stochastic GD)
    Returns:
        [eigenvectors, eigenvalue, model_list]:
                    three lists containing the eigenvectors, eigenvalues, and the corresponding trained solvers.
                    Note that these are returned also in the case of a Keyboardinterrupt

    '''
    n = A.shape[0]

    eigenvectors = []
    eigenvalues = []
    model_list = []
    
    attempt_counter = 0

    train_again = False
    try:
        while len(eigenvalues) < n and attempt_counter<Nattempts:

            print(f"### FINDING EIGENVECTOR NR. {len(eigenvalues)} ###\n")

            #choose a random initial condition to initialize solver
            #train_again is true when user asks for continue training the previous model
            if not train_again:
                starting_point = tf.random.normal([n], dtype='float64')
                solver = esnn.eigSolverNN(A, starting_point)

            #train the model
            solver.train_model(Nepochs, Nbatches, print_info=False)

            #and compute the estimated eigenvector (assuming its solver.x_tilde(solver.t_grid[-1]))
            eigenvalue, eigvector = solver.compute_eig()

            #check if it's an eigenvector
            if check_eig(eigenvalue, eigvector, A):

                #check that the eigenvalue/eigenvector pair hasn't already been found.
                if not np.any(abs(np.array(eigenvalues)-eigenvalue)<0.01):
                    print(f'Found new eigenvector: \n {eigvector} \n eigenvalue {eigenvalue}')
                    #save the eigvalue/eigvec pair, and trained model which is able to reproduce the eigvector
                    eigenvalues.append(eigenvalue[0])
                    eigenvectors.append(eigvector[0])
                    model_list.append(solver)
                else:
                    print(f'Duplicate {eigenvalue}')

                train_again=False
            else:
                print(f'Found something not an eigenvector of A: \n {eigvector}')
                print(f"Do you want to keep training the same model?[y/n]")
                user_in = input()
                if user_in in ['y','Y']:
                    train_again=True
                else:
                    train_again=False

        return eigenvectors, eigenvalues, model_list

    except KeyboardInterrupt:

        return eigenvectors, eigenvalues, model_list

if __name__ == '__main__':
    A = np.load('A.npy')
    A = tf.convert_to_tensor(A)
    #eigenvalues = Many_Random_Points(15, A, 0.0175, tolerance=1e-7, Nepochs=40000)
    eigenvectors, eigenvalues = Search_Until_Find(A)
    print(eigenvalues)
    A = A.numpy()
    [E, V] = np.linalg.eigh(A)
    print(E)
