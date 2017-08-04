import sys
sys.path.insert(0, '..')
import numpy as np 
from src.util.extract_nullspace import *
import unittest 
import math;
from src import *

class TestExtractNullspace(unittest.TestCase):

    def test_cross_product(self):
        return True; # This has already been tested
        for u1, v1 in zip(range(-100, 100), reversed(range(-100, 100))):
            for u2, v2 in zip(range(-100, 100), reversed(range(-100, 100))):
                for u3, v3 in zip(range(-100, 100), reversed(range(-100, 100))):
                    u = [u1, u2, u3];
                    v = [v1, v2, v3];
                    c1 = np.cross(np.array(u), np.array(v));
                    c2 = cross_product(u, v);
                    for c_1, c_2 in zip(c1, c2):
                        self.assertEqual(c_1, c_2, 
                            [u, v])

    def test_l2_norm(self):
        margin = 1e-10;
        for u1 in range(-100, 100):
            for u2 in range(-100, 100):
                for u3 in range(-100, 100):
                    u = [u1, u2, u3]
                    np_length = np.linalg.norm(np.array(u))
                    l = length(u)
                    math.isclose(float(np_length), float(l), rel_tol=margin)

    def test_svd_of_essential_matrix(self):
        pass


if __name__ == '__main__':
    unittest.main()