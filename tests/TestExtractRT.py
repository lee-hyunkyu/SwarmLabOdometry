import sys
sys.path.insert(0, '..')
import numpy as np 
from src.util.extract_R_t import *
import unittest 
import math;
from src import *
import pickle;
import numpy.testing as nptest
import logging

log_file_path = '../logs/test.text'
logging.basicConfig(filename    = log_file_path, 
                    level       = logging.DEBUG,
                    filemode    = 'w',
                    format      = '%(message)s')
logger = logging.getLogger()

class TestExtractRT(unittest.TestCase):

    def test_cross_product(self):
        for _ in range(100):
            u = np.random.rand(3)
            v = np.random.rand(3)
            c1 = np.cross(u, v)
            u = u.tolist()
            v = v.tolist()
            c2 = cross_product(u, v)
            nptest.assert_array_almost_equal(c1, np.array(c2))
                    
    def test_l2_norm(self):
        margin = 1e-10;
        for _ in range(100):
            u = np.random.rand(3)
            np_length = np.linalg.norm(u)
            u = u.tolist()
            l = length(u)
            nptest.assert_almost_equal(np_length, l)

    def test_svd_of_essential_matrix(self):
        S = np.diag([1, 1, 0])
        for i in range(2, 250):
            with open('saved_test_data/essential_matrix/essential_matrix_{:06d}.p'.format(i), 'rb') as f:
                E = pickle.load(f)
                # import pdb; pdb.set_trace();
                # U, s, V = np.linalg.svd(E, full_matrices=True)
                # logger.info('E=' + str(E) + '\nU=' + str(U) + '\ns=' + str(s) + '\nV=' + str(V) + '\n=====>')
                U, V, scale = scaled_svd(E)
                U = np.array(U)
                V = np.array(V)
                E_ = np.dot(U, np.dot(S, np.transpose(V)))
                nptest.assert_array_almost_equal(E, scale*np.array(E_))

    def test_matrix_multiplication(self):
        for _ in range(1000): # Test 100 times
            mat     = np.random.rand(3, 3)
            u       = np.random.rand(3)
            np_p    = np.dot(mat, u)
            mat     = mat.tolist()
            u       = u.tolist()
            p       = dot(mat, u)
            nptest.assert_array_almost_equal(np_p, np.array(p))

    def test_dot_product(self):
        for _ in range(100):
            u = np.random.rand(3)
            v = np.random.rand(3)
            d1 = np.dot(u, v);
            u = u.tolist()
            v = v.tolist()
            d2 = dot_v(u, v);
            nptest.assert_array_almost_equal(d1, np.array(d2))

    @unittest.skip
    def test_mat(self):
        for _ in range(100):
            A = np.random.rand(3,3)
            B = np.random.rand(3,3)
            np_prod = np.dot(A, B)
            A = A.tolist()
            B = B.tolist()
            prod = dot_mat(A, B)
            nptest.assert_array_almost_equal(np_prod, np.array(prod))

    def test_transpose(self):
        for _ in range(100):
            A = np.random.rand(3, 3)
            np_transpose = np.transpose(A)
            trans = transpose(A.tolist())
            nptest.assert_array_equal(np_transpose, trans)

    def test_R_t(self):
        pass

if __name__ == '__main__':
    unittest.main()