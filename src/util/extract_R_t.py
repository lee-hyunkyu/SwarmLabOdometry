import math

def extract_R_t(E):
    ''' Given a matrix E, recovers R, t '''
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
    u1, u2, u3 = u;
    return math.sqrt(dot_v(u, u));
    
def dot_v(u, v):
    ''' Returns the dot product of two 3x1 vectors '''
    u1, u2, u3 = u;
    v1, v2, v3 = v;
    return u1*v1 + u2*v2 + u3*v3

def dot(mat, u):
    ''' Returns the matrix multiplication of a 3x3 mat,  3x1 u '''
    m1, m2, m3 = mat; # Get the row vectors of mat;
    return [ dot_v(m1, u), dot_v(m2, u), dot_v(m3, u) ]

def multiply_scaler(u, s):
    ''' Returns product of some 3x1 vector u with a scaler s '''
    u1, u2, u3 = u;
    u1, u2, u3 = (s*u1, s*u2, s*u3)
    return [u1, u2, u3]

def dot_mat(A, B):
    ''' Returns a matrix that is the multiplication of 2 R(3x3) matrixes '''
    # First transpose B in order to get the columns of B
    pass

def transpose(A):
    '''
    Returns the transpose of a 3x3 matrix
    '''
    a1, a2, a3 = A # Get the row vectors

    # Decompose the row vectors
    a11, a12, a13 = a1
    a21, a22, a23 = a2
    a31, a32, a33 = a3

    # Build the new row vectors
    a1 = [a11, a21, a31]
    a2 = [a12, a22, a32]
    a3 = [a13, a23, a33]

    return [a1, a2, a3]

def scaled_svd(E):
    ''' Returns an SVD of an essential matrix E such that E ~ USV*
        The multiplication is only correct up to a scale
        Assumes that E is of the form [ (a b c), (d e f), (g h i) ]
        Assumes that E is a true Essential matrix, i.e. rank 2 
    '''
    e_a, e_b, e_c = E
    c1 = cross_product(e_a, e_b)    
    c2 = cross_product(e_a, e_c)
    c3 = cross_product(e_b, e_c)
    l1, l2, l3 = (length(c1), length(c2), length(c3))
    longestLength = max(l1, l2, l3)

    if l1 is longestLength:
        v_c = multiply_scaler(c1, 1/length(c1))
    elif l2 is longestLength:
        v_c = multiply_scaler(c2, 1/length(c2))
    else: 
        v_c = multiply_scaler(c3, 1/length(c3))

    v_a = multiply_scaler(e_a, 1/length(e_a))
    v_b = cross_product(v_c, v_a)

    # Properly construct V from the column vectors v_a, v_b, v_c
    v11, v21, v31 = v_a
    v12, v22, v32 = v_b
    v13, v23, v33 = v_c
    V = [(v11, v12, v13), (v21, v22, v23), (v31, v32, v33)]

    # Calculate columns of U from the vectors v
    u_a = dot(E, v_a)
    scaling_factor = length(u_a); # This is for testing purposes
    u_a = multiply_scaler(u_a, 1/length(u_a))

    u_b = dot(E, v_b)
    scaling_factor += length(u_b)
    u_b = multiply_scaler(u_b, 1/length(u_b))

    u_c = cross_product(u_a, u_b)
    u_c = multiply_scaler(u_c, 1/length(u_c))

    # Properly construct U from the column vectors u_a, u_b, u_c
    u11, u21, u31 = u_a
    u12, u22, u32 = u_b
    u13, u23, u33 = u_c
    U = [(u11, u12, u13), (u21, u22, u23), (u31, u32, u33)]

    scaling_factor = scaling_factor/2 # Takes the average because of floating point errors

    return (U, V, scaling_factor)
