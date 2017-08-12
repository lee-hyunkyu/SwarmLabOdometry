'''
Implementation of Nister's 5 point algorithm without any dependencies
'''

import math
import logging
log_file_path = '../logs/test.text'
logging.basicConfig(filename    = log_file_path, 
                    level       = logging.DEBUG,
                    filemode    = 'w',
                    format      = '%(message)s')
logger = logging.getLogger()

def extract_R_t(E, prev_pts, curr_pts, principal_point, focal_length):
    ''' Given a matrix E, recovers R, t '''
    U, V, _ = scaled_svd(E)
    u1, u2, u3 = transpose(U)
    Vt = transpose(V)
    v1, v2, v3 = Vt # Get the columns of V

    t = u3

    D = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    R_a = dot_mat(U, dot_mat(D, Vt))
    PA, PB, PC, PD = 0, 0, 0, 0 # Initialize 
    t1, t2, t3 = t
    # Construct the pose [R_a | t]
    R1, R2, R3 = R_a
    potential_pose = [  [R1[0], R1[1], R1[2], t1], 
                        [R2[0], R2[1], R2[2], t2],
                        [R3[0], R3[1], R3[2], t3] ]
    R_b = dot_mat(U, dot_mat(transpose(D), Vt))

    # Prepare for the loop below that will look at 100 points to determine the correct camera matrix
    dist = 3000  # filter out points that are too far away
    pp_x, pp_y = principal_point
    # Construct Ht, a matrix that transfrom P_a -> P_c
    Ht = [  [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-2*v3[0], -2*v3[1], -2*v3[2], -1]  ]
    total = len(prev_pts)
    for i in range(10):
        # Convert my points and change the origin 
        q0 = prev_pts[total%(i+1)]
        q1 = curr_pts[total%(i+1)]
        # "Multiply" my points by the inverse of intrinsic camera matrix
        q0 = [q0[0], q0[1], 1]
        q1 = [q1[0], q1[1], 1]
        q0 =    [q0[0]/focal_length - pp_x/focal_length, \
                 q0[1]/focal_length - pp_y/focal_length, \
                 1 ] 
        q1 =    [q1[0]/focal_length - pp_x/focal_length,\
                 q1[1]/focal_length - pp_y/focal_length,\
                 1 ] 
        
        if i == 0:
            import numpy as np
            logger.info('============')
            logger.info('qEq = {:f}'.format(np.dot(q1, np.dot(E, q0))))
            logger.info('det(U) = {:f}'.format(np.linalg.det(U)))
            logger.info('det(V) = {:f}'.format(np.linalg.det(V)))
        # Triangulate and find Real World Coordinates
        Q = triangulation(E, q0, q1, potential_pose)
        Q1, Q2, Q3, Q4 = Q
        # Cheirality Check; Make sure the point Q is in front of both cameras
        c1 = Q3*Q4
        c2 = dot(potential_pose, Q)[2]*Q4   

        # filter out far away points (infinite points)
        # At that distance, depth varies between positive and negative
        if abs(Q3/Q4) < dist: 
            if c1 > 0 and c2 > 0:
                PA = PA + 1
            elif c1 < 0 and c2 < 0:
                PB = PB + 1
            else:
                HtQ = dot(Ht, Q)
                if HtQ[3]*Q3 > 0:
                    PC = PC + 1
                else:
                    PD = PD + 1
    
    mostLikelyPose = max(PA, PB, PC, PD)
    logger.info("PA = {:2d},\tPB = {:2d},\tPC = {:2d},\tPD = {:2d},\tsum = {:2d}".format(PA, PB, PC, PD, PA+PB+PC+PD))
    if mostLikelyPose == PA:
        logger.info('PA')
        return R_a, t
    elif mostLikelyPose == PB:
        logger.info('PB')
        return R_a, t
        return R_a, multiply_scaler(t, -1)
    elif mostLikelyPose == PC:
        logger.info('PC')
        return R_b, t
    else:
        logger.info('PD')
        return R_b, t
        return R_b, multiply_scaler(t, -1)
    return R, t

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
    ''' Returns the dot product of two vectors '''
    l1, l2 = (len(u), len(v))
    if l1 is not l2:
        dot_prod = None # Invalid
    else:
        dot_prod = 0
        for i in range(l1):
            dot_prod += u[i]*v[i]
    return dot_prod

def dot(mat, u):
    ''' Returns the matrix multiplication of a nx3 mat,  3x1 u '''
    product = []
    for i in range(len(mat)):
        product += [dot_v(mat[i],u)]
    return product

def multiply_scaler(u, s):
    ''' Returns product of some vector u with a scaler s '''
    l1 = len(u)
    new_vec = [0 for i in range(l1)]
    for i in range(l1):
        new_vec[i] = u[i]*s
    return new_vec  

def dot_mat(A, B):
    ''' Returns a matrix that is the multiplication of 2 R(3x3) matrixes '''
    # First transpose B in order to get the columns of B
    n1 = len(A) # Get the number of rows of A
    n2 = len(B) # Get the number of rows of B
    B_t = transpose(B)
    # Get the number of columns
    m2 = len(B)
    m1 = len(A[0]) # Assumes that the matrix is properly formed

    # Check that matrix multiplication is valid
    if n1 is not m2 and n2 is not m1:
        return None

    new_mat = []
    for i in range(m2):
        new_mat += [dot(A, B_t[i])]

    # Transpose
    return transpose(new_mat)

def transpose(A):
    '''
    Returns the transpose of a matrix
    '''
    m1, n1 = len(A), len(A[0])
    new_mat = [[] for i in range(n1)]
    for i in range(n1):
        for j in range(m1):
            new_mat[i] += [A[j][i]]
    return new_mat

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

def triangulation(E, q0, q1, potential_pose):
    ''' Given the essential matrix, 2 corresponding points, and a potential solution of R, t
        q0 is the point in view 1, q1 is hte same point in view 2
    '''
    diag = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    a = dot(transpose(E), q1)
    b = cross_product(q0, dot(diag, a))
    d = cross_product(a, b)
    # Convert to homogenous coordinate
    # d = [d[0]/d[2], d[1]/d[2], 1] 
    Q = [0 for i in range(4)] # initialize with 0s
    #assert(d[0]*q0[0] > 0 and d[1]*q0[1] > 0) # Make assumption that they're in the same ball park
    
    c = cross_product(q1, dot(diag, dot(E, q0)))
    C1, C2, C3, C4 = dot(transpose(potential_pose), c)
    Q1, Q2, Q3 = multiply_scaler(d, C4)

    #import pdb; pdb.set_trace()
    Q4 = dot_v(d, [C1, C2, C3])
    Q = [Q1, Q2, Q3, -Q4]
    import numpy as np
    return Q 
