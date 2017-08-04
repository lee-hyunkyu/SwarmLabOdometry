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

    @unittest.skip
    def test_cross_product(self):
        for u1, v1 in zip(range(-100, 100), reversed(range(-100, 100))):
            for u2, v2 in zip(range(-100, 100), reversed(range(-100, 100))):
                for u3, v3 in zip(range(-100, 100), reversed(range(-100, 100))):
                    u = [u1, u2, u3];
                    v = [v1, v2, v3];
                    c1 = np.cross(np.array(u), np.array(v));
                    c2 = cross_product(u, v);
                    
    @unittest.skip
    def test_l2_norm(self):
        margin = 1e-10;
        for u1 in range(-100, 100):
            for u2 in range(-100, 100):
                for u3 in range(-100, 100):
                    u = [u1, u2, u3]
                    np_length = np.linalg.norm(np.array(u))
                    l = length(u)
                    self.assertTrue(math.isclose(float(np_length), float(l), rel_tol=margin))

    def test_svd_of_essential_matrix(self):
        pass

    def test_matrix_multiplication(self):
        pass

    def test_dot_product(self):
        margin = 1e-10;
        for u1, v1 in zip(range(-30, 30), reversed(range(-30, 30))):
            for u2, v2 in zip(range(-30, 30), reversed(range(-30, 30))):
                for u3, v3 in zip(range(-30, 30), reversed(range(-30, 30))):
                    u = [u1, u2, u3];
                    v = [v1, v2, v3];
                    d1 = np.dot(np.array(u), np.array(v))
                    d2 = dot_v(u, v);
                    self.assertEqual(d1, d2)

if __name__ == '__main__':
    unittest.main()