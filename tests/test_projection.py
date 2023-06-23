import unittest
import pytest
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from utility_functions.canonical_simplex import simplex_projection_1d, simplex_projection, \
    feasible_left_to_right_projection


@pytest.mark.tm
class TestCanonicalSimplex(unittest.TestCase):
    def setUp(self):
        self.one_d = np.array([[0., 0., 0., 1.],
                               [0., 0., 1., 1e-5],
                               [0.1, 0.1, 0.8, 0.1],
                               [0., 0.8, 0.1, 0.1],
                               np.log([0.1, 0.1, 0.8, 0.1])]).astype(np.float32)
        self.matrices = [np.array([[0., 1., 0., 0.],
                                   [0., 0.875165939, 0.124834061, 0.],
                                   [0., 0., 0.80000031, 0.19999969],
                                   [0., 0.044600293, 0., 0.955399692]]).astype(np.float32),
                         np.array([[.8, .2, 0., 0.],
                                   [0., .7, .3, 0.],
                                   [0., 0., .5, .5],
                                   [0.1, 0., 0., 0.9]]).astype(np.float32),
                         np.array([[0., 1., 0., 0.],
                                   [0., .4, .5, 1e-5],
                                   [0., 0., .5, .5],
                                   [0.0, 0., 0., 1.]]).astype(np.float32),
                         np.array([[.8, .2, 2., 3.],
                                   [-.1, .7, .3, 0.],
                                   [-.1, -.1, .2, .2],
                                   [0.1, 0., 0., 0.9]]).astype(np.float32)
                         ]

    def test_vec_in_simplex(self):
        """
        Tests if L2 norm of the vector is in canonical simplex.
        For a vector of floats, this defines a density estimation.
        """
        for vec in self.one_d:
            proj = simplex_projection(vec)
            npt.assert_almost_equal(np.sum(proj), 1.0)
            assert np.all(proj >= 0)
        for mat in self.matrices:
            for col in mat:
                proj = simplex_projection(col)
                npt.assert_almost_equal(np.sum(simplex_projection(col)), 1.0)
                assert np.all(proj >= 0)

    def test_simplex_1d(self):
        # identity if b in Pb
        exp_output = np.array([0.5, 0.5, 0, 0]).astype(np.float32)
        mat = np.array([0.5, 0.5, 0, 0]).astype(np.float32)
        projection = simplex_projection_1d(mat).reshape((4,))
        npt.assert_almost_equal(projection, exp_output)

        projection = simplex_projection_1d(np.array([0.01, 0.9, 0])).reshape((3,))
        npt.assert_almost_equal(projection, np.array([0.04, 0.93, 0.03]))

        projection = simplex_projection_1d(np.array([0.01, 0.9, -0.1])).reshape((3,))
        npt.assert_almost_equal(projection, np.array([0.0550, 0.945, 0.0]))

        projection = simplex_projection_1d(np.array([0.9, 0.5, 0])).reshape((3,))
        npt.assert_almost_equal(projection, np.array([0.7, 0.3, 0.0]))

        projection = simplex_projection_1d(np.array([0.5, 0.9, 0])).reshape((3,))
        npt.assert_almost_equal(projection, np.array([0.3, 0.7, 0.0]))

        projection = simplex_projection_1d(np.array([0.5, 0.9])).reshape((2,))
        npt.assert_almost_equal(projection, np.array([0.3, 0.7]))

        projection = simplex_projection_1d(np.array([0.9, 0.5])).reshape((2,))
        npt.assert_almost_equal(projection, np.array([0.7, 0.3]))



    def test_identity_simplex(self):
        exp_output = np.array([[0.5, 0.5, 0, 0],
                               [0, 0.5, 0.5, 0],
                               [0, 0, 0.5, 0.5],
                               [0.5, 0, 0, 0.5]]).astype(np.float32)
        mat = np.array([[0.5, 0.5, 0, 0],
                        [0, 0.5, 0.5, 0],
                        [0, 0, 0.5, 0.5],
                        [0.5, 0, 0, 0.5]]).astype(np.float32)

        proj = simplex_projection(mat)
        npt.assert_equal(proj, exp_output)

    def test_left_to_right_adjustment(self):
        """
        Tests if transition matrix defines a left to right HMM,
        with densities only in adjacent states in the underlying target model
        """
        exp_output = np.array([[0.5, 0.5, 0, 0],
                               [0, 0.5, 0.5, 0],
                               [0, 0, 0.5, 0.5],
                               [0.5, 0, 0, 0.5]]).astype(np.float32)
        mat = np.array([[0.5, 0.5, 0, 0],
                        [0, 0.5, 0.5, 0],
                        [0, 0, 0.5, 0.5],
                        [0.5, 0, 0, 0.5]]).astype(np.float32)

        proj = feasible_left_to_right_projection(mat)
        npt.assert_equal(proj, exp_output)

        mat = np.array([[0.2, 0.8, 0, 0], [0, 0.7, 0.3, 0], [0, 0, 0.1, 0.9], [0.3, 0, 0, 0.7]])
        proj = feasible_left_to_right_projection(mat)
        npt.assert_equal(proj, mat)
