#import eigenTest as et
import numpy as np

def normalized_orthagonal_part(vector1, vector2):
    normalization = np.dot(vector1, vector2)/np.dot(vector2, vector2)
    return normalization*vector2

def create_orthogonal(vectors_array):
    """
    Creates a random vector and substracts all orthogonal parts with vectors
    from vector_array
    """
    n = len(vectors_array[0]) #size of vectors
    orthogonal_vector = np.random.randn(n)

    # Gram-Schmidt algorythm
    for vector in vectors_array:
        orthogonal_vector -= normalized_orthagonal_part(orthogonal_vector, vector)

    return orthogonal_vector


def FindAllEigenvectors(Matrix_to_diagonalize):
    n = len(Matrix_to_diagonalize)
    eigenvectors = np.zeros((n,n))
    eigenvalues = np.zeros(n)

    starting_point = np.random.randn(n)

    for i in range(n):
        eigenvector, eigenvalue = et.FindEigs(starting_point)
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
