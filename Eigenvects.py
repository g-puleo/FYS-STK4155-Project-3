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
    Note that this will be the 0 vector if there are more than n vectors in vectors_array .
    """
    n = len(vectors_array[0]) #size of vectors
    orthogonal_vector = np.random.randn(n)

    # Gram-Schmidt algorithm
    for vector in vectors_array:
        orthogonal_vector -= normalized_proj(orthogonal_vector, vector)

    return orthogonal_vector

def check_eig(eigval, eigvec, A):
    '''check that eigvec is an eigenvector of A with eigenvalue eigval.
    All args must be given as tf.Tensor() instances, and are converted to numpy here.
    '''

    eigval = eigval.numpy()
    eigvec = eigvec.numpy()
    A = A.numpy()

    #lambdavec should have components all equal to eigval
    lambdavec = (A@eigvec)/eigvec
    #dividing by eigval should give a vector of ones
    ones_vec = lambdavec/eigval
    #check that this vector is close enough to np.ones()
    A = np.allclose(ones_vec, np.ones(ones_vec.shape), rtol=1e-2)
    if not A:
        warnings.warn(f"eigenvector ({eigvec}) might not be an eigenvector of A.")


def findEigenvectors(A):
    '''
    Find all eigenvectors of matrix A, using neural net.

    '''
    n = A.shape[0]

    eigenvectors = tf.zeros((n,n))
    eigenvalues = tf.zeros(n)

    starting_point = tf.random.normal(n)

    #initialize instance of solver
    Nepochs = 10000
    Nbatches = 10 

    for i in range(n):
        print(f"### FINDING EIGENVECTOR NR.{i} ###\n")
        solver = esnn.eigSolverNN(A, starting_point)
        solver.train_model(Nepochs, Nbatches)
        eigenvalue, eigvector = esnn.compute_eig()

        #SHOULD CHECK THAT WHAT WE HAVE GOTTEN SO FAR *IS* AN EIGENVECTOR.
        #OTHERWISE, TRAINING AGAIN WOULD NOT MAKE SENSE
        check_eig(eigenvalue, eigvector, A)


        eigenvectors[i, :] = eigenvector
        eigenvalues[i] = eigenvalue
        starting_point = create_orthogonal(eigenvectors[:i+1, :])

    return eigenvectors, eigenvalues

if __name__ == '__main__':
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    vectors = np.array([v1, v2])
    u2 = create_orthogonal(vectors)
    print(np.dot(v1, u2), np.dot(v2, u2), u2)
