import tensorflow as tf


def get_averaged_predictions(target, y_pred, patch_size=64, stride=8, num_classes=4):
    """
    Computes the average predictions per overlapping windows

    Parameters
    ----------
    y_true : tf.Tensor
        The tensor of one-hot-encoded ground-truth
    y_pred : tf.Tensor
        The tensor of the output layer of a DNN

    Returns
    -------
    tf.Variable
        The negative MMI value for optimization
    """
    length = tf.shape(target)[0]
    # Compute the indices were the windows start and end
    start = tf.range(start=0, limit=length - patch_size + 1, delta=stride)
    end = tf.range(start=patch_size, limit=length + 1, delta=stride)
    cum_sum = tf.zeros(shape=[length, num_classes], dtype=tf.float32)
    overlapping_windows = tf.zeros(shape=[length], dtype=tf.float32)
    for i in range(tf.shape(start)[0]):
        # pad the indices that are not in the window and add the window value to the sum
        cum_sum = cum_sum + tf.cast(tf.pad(y_pred[i], paddings=[[start[i], length - end[i]], [0, 0]]),
                                    dtype=tf.float32)
        # add a mask of LENGTH zeros with PATCH_SIZE ones where the window is defined
        # thus memorizing the number of windows that contribute to a prediction
        overlapping_windows = overlapping_windows + tf.pad(
            tf.ones(shape=[patch_size]), paddings=[[start[i], length - end[i]]]
        )
    cond = (length - patch_size) % stride > 0  # condition for squeezing the last window
    if cond:
        last_window_start, last_window_end = length - patch_size, length
        start = tf.pad(start, [[0, 1]], constant_values=last_window_start)
        end = tf.pad(end, [[0, 1]], constant_values=last_window_end)
        idx = tf.shape(start)[0] - 1
        cum_sum = cum_sum + tf.cast(tf.pad(y_pred[idx], paddings=[[start[idx], length - end[idx]], [0, 0]]),
                                    dtype=tf.float32)
        overlapping_windows = overlapping_windows + tf.pad(
            tf.ones(shape=[patch_size]), paddings=[[start[idx], length - end[idx]]]
        )

    # Compute the final average. Transpose operations used to make it broadcastable
    logits = tf.transpose(tf.transpose(cum_sum) / overlapping_windows)
    return logits