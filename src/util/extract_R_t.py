import math

def extract_R_t(E, prev_pts, curr_pts):
    ''' Given a matrix E, recovers R, t '''
    U, V, _ = scaled_svd(E)
    u1, u2, u3 = transpose(U) # Get the columns of U
    v1, v2, v3 = transpose(V) # Get the columns of V
    t = multiply_scaler(u3, 1/length(u3)) # Normalize to be of length 1
    D = [[1, 0, 0], [0, 1, 0], [0, 0, 0]] # diag(1, 1, 0)
    R_a = dot_mat(U, dot_mat(D, transpose(V)))

    # Construct P_A = [ R_a | t]
    t1, t2, t3 = t
    R1, R2, R3 = R_a
    r11, r12, r13 = R1 
    r21, r22, r23 = R2 
    r31, r32, r33 = R3 

    P_A = [ [r11, r12, r13, t1],
            [r21, r22, r23, t2], 
            [r31, r32, r33, t3] ]

    pp1, pp2 = prev_pts[0]
    cp1, cp2 = curr_pts[0]

    Q = triangulation(E, [pp1, pp2, 1], [cp1, cp2, 1], P_A)

    # Determine the real location
    # Must be in front of both cameras
    isInFrontOfFirstCamera     = False
    isInFrontOfSecondCamera    = False
    Q1, Q2, Q3, Q4 = Q
    c1 = Q3*Q4 
    import pdb; pdb.set_trace()
    c2 = dot_v(P_A, Q)
    c2 = c2[3]*Q4

    pose = P_A

    if c1*c2 < 0:
        # Rotation is incorrect. Rotate another 180 around the baseline
        R_b = dot_mat(U, dot_mat(transpose(D), transpose(V)))
        R1, R2, R3 = R_b
        pose = P_A
        
        # Replace R_a with R_b
        # TODO: Write a better matrix multiplication algorithm
        pose[0][0] = R1[0]; pose[0][1] = R1[1]; pose[0][2] = R1[2]
        pose[1][0] = R2[0]; pose[0][1] = R2[1]; pose[0][2] = R2[2]
        pose[2][0] = R3[0]; pose[0][1] = R3[1]; pose[0][2] = R3[2]

        # The translaformation that gives us a "twisted pair"
        Ht = [  [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-2*v3[0], -2*v3[1], -2*v3[2], -1]
             ]

        HtQ = dot_v(Ht, Q)
        reverse_translation = Q3*HtQ[3] < 0
    else:
        # Rotation is R_a
        reverse_translation = c1*c2 > 0

    if reverse_translation:
        # Reverse translation; This is the case of inverting the last column of the pose
        pose = P_A
        pose[0][3] = -pose[0][3]
        pose[1][3] = -pose[1][3]
        pose[2][3] = -pose[2][3]

    return pose, t

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
    ''' Returns the matrix multiplication of a nx3 mat,  3x1 u '''
    product = []
    for i in range(len(mat)):
        product += [dot_v(mat[i],u)]
    return product

def multiply_scaler(u, s):
    ''' Returns product of some 3x1 vector u with a scaler s '''
    u1, u2, u3 = u;
    u1, u2, u3 = (s*u1, s*u2, s*u3)
    return [u1, u2, u3]

def dot_mat(A, B):
    ''' Returns a matrix that is the multiplication of 2 R(3x3) matrixes '''
    # First transpose B in order to get the columns of B
    B_t = transpose(B)
    b1, b2, b3 = B_t # b1, b2, b3 are the column vectors of B

    # Create the column vectors for the product
    a1, a2, a3 = dot(A, b1), dot(A, b2), dot(A, b3)
    # Stack them as row vectors
    C = [a1, a2, a3]
    # Transpose
    return transpose(C)

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
    V = transpose([v_a, v_b, v_c])

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
    U = transpose([u_a, u_b, u_c])

    scaling_factor = scaling_factor/2 # Takes the average because of floating point errors

    return (U, V, scaling_factor)

def triangulation(E, q0, q1, potential_P):
    ''' Given the essential matrix, 2 corresponding points, and a potential solution of R, t
        q0 is the point in view 1, q1 is hte same point in view 2
    '''
    diag = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    c = cross_product(q1, dot(diag, dot(E, q0)))

    # Transpose the 4x3 matrix potential_P
    P1, P2, P3 = potential_P
    p11, p12, p13, p14 = P1 
    p21, p22, p23, p24 = P2 
    p31, p32, p33, p34 = P3 

    potential_P_t = [ [p11, p21, p31],
                      [p12, p22, p32],
                      [p13, p23, p33], 
                      [p14, p24, p34] ]

    C = dot(potential_P_t, c)
    C1, C2, C3, C4 = C
    d1, d2, d3 = q0
    Q = [0, 0, 0, 0] # placeholder
    Q[0] = d1*C4
    Q[1] = d2*C4
    Q[2] = d3*C4
    Q[3] = -dot_v([d1, d2, d3], [C1, C2, C3])
    return Q

