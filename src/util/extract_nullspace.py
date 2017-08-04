def extract_nullspace(mat):
    ''' Assumes that mat is a valid essential matrix
        Determines the nullspace of mat given the above assumption.
        Implementation of Nister's Efficient SVD of Essential  Matrix 
    '''
    pass

def cross_product(u, v):
    ''' Returns (u x v) '''
    u1, u2, u3 = u
    v1, v2, v3 = v
    c1 = u2*v3 - u3*v2
    c2 = -(u1*v3-u3*v1)
    c3 = u1*v2 - u2*v1
    return [c1, c2, c3]

def length(u):
    ''' Returns the L2 norm of a 3d vector u '''
    pass

def dot(mat, u):
    ''' Returns the matrix multiplication of a 3x3 mat,  3x1 u '''
    pass

def svd(E):
    ''' Returns the SVD of an essential matrix E'''
    ''' Assumes that E is of the form [ (a b c), (d e f), (g h i) ]'''
    ''' Assumes that E is a true Essential matrix, i.e. rank 2 '''