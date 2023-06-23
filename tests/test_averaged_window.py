import unittest
import pytest
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from scipy.special import softmax

from models.processing_layers import get_averaged_predictions


@pytest.mark.tm
class TestAverageWindow(unittest.TestCase):

    def test_averaged_windows_offset(self):
        T = 100
        patch_size = 64
        stride = 8
        num_classes = 4

        number_windows = int((T - patch_size) / stride) + 1
        if (T - patch_size) % stride > 0:
            number_windows = int(round(number_windows + 1))
        y_pred_offset = softmax(
            np.random.uniform(size=(number_windows, patch_size, 4), low=0, high=1),
            axis=2)
        y_true_offset = softmax(
            np.random.uniform(size=(T, patch_size, 4), low=0, high=1),
            axis=2)
        windowed_predictions = get_averaged_predictions(target=y_true_offset,
                                                        y_pred=y_pred_offset,
                                                        patch_size=patch_size,
                                                        stride=stride,
                                                        num_classes=num_classes)

        sum_windowed_predictions = np.sum(windowed_predictions, axis=1)

        npt.assert_almost_equal(np.ones(y_true_offset.shape[0]), sum_windowed_predictions)

    def test_averaged_windows_no_offset(self):
        T = 64
        patch_size = 64
        stride = 8
        num_classes = 4

        number_windows = int((T - patch_size) / stride) + 1
        if (T - patch_size) % stride > 0:
            number_windows = int(round(number_windows + 1))
        y_pred_offset = softmax(
            np.random.uniform(size=(number_windows, patch_size, 4), low=0, high=1),
            axis=2)
        y_true_offset = softmax(
            np.random.uniform(size=(T, patch_size, 4), low=0, high=1),
            axis=2)
        windowed_predictions = get_averaged_predictions(target=y_true_offset,
                                                        y_pred=y_pred_offset,
                                                        patch_size=patch_size,
                                                        stride=stride,
                                                        num_classes=num_classes)

        sum_windowed_predictions = np.sum(windowed_predictions, axis=1)

        npt.assert_almost_equal(np.ones(y_true_offset.shape[0]), sum_windowed_predictions)
