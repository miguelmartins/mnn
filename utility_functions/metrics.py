from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score


def get_metrics(gt, prediction):
    ppv, sensitivity, accuracy = [], [], []

    for sound in range(len(gt)):
        ppv_, sensitivity_ = schmidt_metrics(gt[sound], prediction[sound])
        accuracy.append(accuracy_score(gt[sound], prediction[sound]))
        ppv.append(ppv_)
        sensitivity.append(sensitivity_)
    mean_ppv = np.mean(ppv)
    mean_sensitivity = np.mean(sensitivity)
    mean_accuracy = np.mean(accuracy)
    return mean_ppv, mean_sensitivity, mean_accuracy


def get_segments(y: np.ndarray) -> np.ndarray:
    """
    Given a 1D state sequence, determines the start and end segment number for each
    contiguous state segment
    Parameters
    ----------
    y: a 1d np.ndarray containing a state sequence

    Returns
    -------
    A N_soundsX 2 X 3 matrix of the form A_i = [start_i, end_i, state_i]
    """
    segments = []
    signal_length = y.shape[0]
    start = 0
    for i in range(1, signal_length):
        if y[i] != y[i - 1]:
            segments.append([start, i - 1, y[i - 1]])
            start = i

    segments.append([start, signal_length - 1, y[-1]])
    return np.array(segments)


def get_centers(segments: np.ndarray) -> np.ndarray:
    """
    Given the segments start deliniations matrix, computes the center segment for each row.
    Output segment center indices are assumed to be continuous for computational purposes.
    Parameters
    ----------
    segments
        A 2D np.ndarray of the form A_i=[start_i, end_i, state_i]
    Returns
    -------
        A 2D  np.ndarray of the form B_j = [middle_point_j, state_j]
    """
    centers_ = (((segments[:, 1] - segments[:, 0]) / 2) + segments[:, 0])
    return np.stack([centers_, segments[:, 2]]).T  # transpose to get a n_segments X 2 matrix


def count_true_positives(source_sequence: np.ndarray, target_sequence: np.ndarray, threshold: float) -> int:
    """
    Given the two segments compute TP/FP according to [Schmidt08]
    Parameters
    ----------
    target_sequence The reference sequence
    source_sequence The sequence where the TP will be counted
    threshold

    Returns
    -------
        The number of TP of source sequence w.r.t. target_sequence
    """
    true_positives = 0
    for positive in source_sequence:
        positive_center, positive_state = positive[0], positive[1]
        # see where pred and ground and ground truth concur
        join = target_sequence[target_sequence[:, 1] == positive_state]
        # select points that are withing threshold
        candidates = np.where(np.abs(join[:, 0] - positive_center) <= threshold)[0]
        # If the condition is fulfilled at least once, count as tp
        if len(candidates) > 0:
            true_positives += 1
    return true_positives


def compute_tp_fp(true_segment_s: np.ndarray, pred_segment_s: np.ndarray, threshold: float) -> Tuple[int, int]:
    """
    Given the estimates and ground truth for the segments center outputs the number of TP and FP given some
    threshold in time (s)
    according to [Schmidt08]
    Parameters
    ----------
    true_segment_s A 1D array of the form a_i = [center_i state_i]
    pred_segment_s A 1D array of the form a_i = [center_i state_i]
    threshold Time interval to account for TPs (s)

    Returns
    -------
        The number of TP and FP according to [Schmidt08]
    """
    # Filter only positive predictions (S1 [0] and S2 [2])
    true_segment_fundamental = true_segment_s[(true_segment_s[:, 1] == 0) | (true_segment_s[:, 1] == 2)]
    pred_segment_fundamental = pred_segment_s[(pred_segment_s[:, 1] == 0) | (pred_segment_s[:, 1] == 2)]
    tp_ppv = count_true_positives(pred_segment_fundamental, true_segment_fundamental, threshold)
    tp_sensitivity = count_true_positives(true_segment_fundamental, pred_segment_fundamental,  threshold)
    # FP all other sounds that are not TP.
    fp_ppv = len(pred_segment_fundamental) - tp_ppv
    return tp_ppv, fp_ppv, tp_sensitivity


def get_schmidt_tp_fp(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sample_rate: int = 50,
                      threshold: float = 0.06) -> Tuple[int, int]:
    """
    Computes TP and FP for PCG pairs (y_true, y_pred), where S1 (state == 0) and S2 (state == 2)
    are assumed to be the positive predictions. See [Schmidt08] for details. Summary:
    A true positive is considered if the distance to the mid point of each segment in y_true and y_pred
    is less than a certain threshold (default 60 ms). All other instances are considered FP.
    Parameters
    ----------
    y_true The ground truth sequence
    y_pred The prediction sequence
    sample_rate The signal sample rate
    threshold The TP window threshold (defaults to 60 ms)

    Returns
    -------
        (int, int, int) A tuple containing the #tp and #fp and total (# of s1 and s2 in ground truth) respectively.
    """

    # Get center segments
    true_segment_s = get_centers(get_segments(y_true))
    pred_segment_s = get_centers(get_segments(y_pred))
    # Convert to time domain (seconds)
    true_segment_s[:, 0] = true_segment_s[:, 0] / sample_rate
    pred_segment_s[:, 0] = pred_segment_s[:, 0] / sample_rate
    # Determine tp and fp within tolerance threshold
    tp_ppv, fp_ppv, tp_sens = compute_tp_fp(true_segment_s, pred_segment_s, threshold)
    # Find s1 and s2 segments
    true_segment_s1 = true_segment_s[true_segment_s[:, 1] == 0]
    true_segment_s2 = true_segment_s[true_segment_s[:, 1] == 2]
    total = len(true_segment_s1) + len(true_segment_s2)  # Number of S1 and S2 in Ground truth (as per [Renna17])
    return tp_ppv, fp_ppv, total, tp_sens


def schmidt_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    sample_rate: int = 50,
                    threshold: float = 0.06) -> Tuple[float, float]:
    tp, fp, total, tp_sens = get_schmidt_tp_fp(y_true, y_pred, sample_rate, threshold)

    try:
        ppv = tp / (tp + fp)
    except:
        ppv = 0.0
    try:
        sensitivity = tp_sens / total
    except:
        sensitivity = 0.0
    return ppv, sensitivity
