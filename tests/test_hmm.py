import unittest
import pytest
import numpy as np
from scripts.synthetic_HMMGMM_vectors import synthetic_HMMGMM_vectors, create_dataset_HMMGMM_vectors
from scripts.custom_functions import *


@pytest.mark.tm
class TestHMM(unittest.TestCase):
    def setUp(self):
        self.trans_mat = np.array([[0.2, 0.8, 0, 0], [0, 0.7, 0.3, 0], [0, 0, 0.1, 0.9], [0.3, 0, 0, 0.7]])
        self.s_emis = [10, 8, 9, 7.5]
        self.m_emis = [2, 0.5, 2, 0.5]
        self.p_states = np.array([0.13, 0.37, 0.12, 0.37])
        self.sigma_noise = 1
        self.Ns = 5
        self.T = 5
        self.nch = 2
        self.sounds = np.array([[[4.38492359],
                                 [-8.17916658]],

                                [[-4.35254787],
                                 [0.61624104]],

                                [[32.51762704],
                                 [-6.2512498]],

                                [[4.6245624],
                                 [-3.05603484]],

                                [[5.63235524],
                                 [-3.06865275]],

                                [[-6.00299461],
                                 [28.37123609]],

                                [[7.79601866],
                                 [-9.35551375]],

                                [[11.6812369],
                                 [-13.03041182]],

                                [[-0.66310697],
                                 [-14.00778332]],

                                [[-3.69112475],
                                 [6.11324356]],

                                [[-11.22457187],
                                 [9.83583672]],

                                [[12.89592016],
                                 [-2.07801908]],

                                [[5.16344845],
                                 [5.25975923]],

                                [[7.34280412],
                                 [1.15387057]],

                                [[8.90711904],
                                 [2.49891827]],

                                [[-1.143772],
                                 [-1.07923935]],

                                [[6.47319465],
                                 [10.70561516]],

                                [[-0.12588053],
                                 [-13.08540352]],

                                [[5.28619255],
                                 [-8.5781063]],

                                [[-0.42232767],
                                 [11.00898343]],

                                [[23.69357199],
                                 [-8.69351337]],

                                [[-5.40553061],
                                 [2.19969002]],

                                [[24.51173768],
                                 [5.50018537]],

                                [[14.11872173],
                                 [14.01120438]],

                                [[-8.16279664],
                                 [-0.49101262]]])
        self.y_true = np.array([[0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [0., 0., 0., 1.],
                                [1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 1., 0., 0.]])
        np.random.seed(42)
        self.y_pred = np.random.permutation(self.y_true)

        self.y_classes = self.y_true.argmax(axis=1)
        self.T = self.y_classes.shape[0]
        self.trans_vec = np.zeros(len(self.y_classes) - 1)
        self.out_vec = np.zeros(len(self.y_classes) - 1)
        self.states_vec = np.zeros(len(self.y_classes) - 1)
        for ind in range(0, len(self.y_classes) - 1):
            self.trans_vec[ind] = self.trans_mat[int(self.y_classes[ind]), int(self.y_classes[ind + 1])]
            self.out_vec[ind] = self.y_pred[ind + 1, int(self.y_classes[ind + 1])]
            self.states_vec[ind] = self.p_states[int(self.y_classes[ind + 1])]

    def test_shape_create_hmm_dataset(self):
        obs, labels = create_dataset_HMMGMM_vectors(self.Ns, self.T, self.nch, self.trans_mat,
                                                    self.m_emis, self.s_emis, self.sigma_noise)
        exp_shape_obs = (self.Ns * self.T, self.nch, 1)
        exp_shape_labels = (self.Ns * self.T, 4)
        # why not obs.shape = (Ns, Ts, nch, 1) ?
        assert (obs.shape == exp_shape_obs) & (labels.shape == exp_shape_labels)

    def test_np_forward_backward(self):
        log_p_os = np.sum(np.log1p(self.trans_vec))+np.sum(np.log1p(self.out_vec))-np.sum(np.log1p(self.states_vec))
        alpha, beta = forward_backward(self.y_true, self.y_pred, self.trans_mat, self.p_states)
        log_p_o = np.log1p(np.sum(alpha[self.T - 1, :]))
        print(alpha, beta)
        print(log_p_o)
        print(log_p_os)

