from random import normalvariate
from math import sqrt
from matrix_operations import *

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    norm = sqrt(sum(x * x for x in unnormalized))
    return [x / norm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = len(A), len(A[0])
    x = randomUnitVector(m)
    lastV = None
    currentV = x

    if n > m:
        B = matmul(transpose(A), A)
    else:
        B = matmul(A, transpose(A))

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = matmul(B, lastV)
        currentVNorm = norm(currentV)
        currentV = [x / currentVNorm for x in currentV]

        if abs(dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    n, m = len(A), len(A[0])
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = [[A[row][col] for row in range(len(A))] for col in range(len(A[0]))]

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * outer(u, v)

        v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = matmul(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    result = transpose(svdSoFar)
    return result[0], result[1], result[2]