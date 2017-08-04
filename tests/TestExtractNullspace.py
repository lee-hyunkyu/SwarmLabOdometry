import sys
sys.path.insert(0, '..')
import numpy as np 
from src.util.extract_nullspace import *
import unittest 
import math;
from src import *
import pickle;
import numpy.testing as nptest

class TestExtractNullspace(unittest.TestCase):

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
        pass

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

if __name__ == '__main__':
    unittest.main()