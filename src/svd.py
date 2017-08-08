from random import normalvariate
from math import sqrt

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

    singularValues, us, vs = transpose(svdSoFar)
    return singularValues, us, vs

def matmul(A, B):
    if len(A[0]) != len(B):
        raise TypeError("Invalid matrix dimensions")
    return [[dot(row, col) for col in transpose(B)] for row in A]

def dot(a, b):
    if isinstance(a, int):
        return [a * x for x in b]
    elif isinstance(b, int):
        return [b * x for x in a]  
    elif len(a) != len(b):
        raise TypeError("Invalid vector dimensions")
    return sum(a[i] * b[i] for i in range(len(a)))

def transpose(M):
    if not isinstance(M[0], list):
        return M
    return [[M[row][col] for row in range(len(M))] for col in range(len(M[0]))]

def norm(M):
    if isinstance(M[0], list):
        x = 0
        for row in range(len(M)):
            for col in range(len(M[row])):
                x += M[row][col]*M[row][col]
        return sqrt(x)
    else:
        return sqrt(dot(M, M))

def outer(a, b):
    return [[i * j for j in b] for i in a]