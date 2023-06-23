import tensorflow as tf
import numpy as np

from models.processing_layers import get_averaged_predictions
from utility_functions.canonical_simplex import simplex_projection, project_matrix_row_components
from utility_functions.hmm_utilities import QR_steady_state_distribution


@tf.function(experimental_relax_shapes=True)
def apply_gradient_nn_only(optimizer, loss_object, model, x, y):
    """
    Calculates and applies the gradient for a given batch and loss object
    only for the DNN parameters given an optimizer

    Parameters
    ----------
    optimizer : tensorflow.keras.optimizers.Optimizer
       An optimizer for the DNN and HMM for Gradient Descent
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    model : tensorflow.python.keras.models.Model
        The DNN model
    x : tf.Tensor
       A tensor containing a batch of observation
    y : tf.Tensor
       A tensor containing the batch ground truth

    Returns
    -------
    tf.Tensor
        The value of the loss function given the method's parameters
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=True)
        loss_value = loss_object(y_true=y, y_pred=logits)

        # Neural Network gradient
        grads = tape.gradient(loss_value, model.trainable_variables)

        # AutoDiff step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return logits, loss_value


@tf.function(experimental_relax_shapes=True)
def apply_gradient(optimizer, loss_object, model, x, y):
    """
    Calculates and applies the gradient for a given batch and loss object
    for the DNN and HMM parameters given an optimizer

    Parameters
    ----------
    optimizer : tensorflow.keras.optimizers.Optimizer
       An optimizer for the DNN and HMM for Gradient Descent
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    model : tensorflow.python.keras.models.Model
        The DNN model
    x : tf.Tensor
       A tensor containing a batch of observation
    y : tf.Tensor
       A tensor containing the batch ground truth

    Returns
    -------
    tf.Tensor
        The value of the loss function given the method's parameters
    """
    with tf.GradientTape() as tape, tf.GradientTape() as p_tape, tf.GradientTape() as t_tape:
        tape.watch(x)
        p_tape.watch(loss_object.p_states)
        t_tape.watch(loss_object.trans_mat)
        logits = model(x, training=True)
        loss_value = loss_object(y_true=y, y_pred=logits)

        # Neural Network gradient
        grads = tape.gradient(loss_value, model.trainable_variables)

        # HMM gradients
        grads_p = p_tape.gradient(loss_value, [loss_object.p_states])
        grads_t = t_tape.gradient(loss_value, [loss_object.trans_mat])

        # AutoDiff step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        optimizer.apply_gradients(zip(grads_p, [loss_object.p_states]))
        optimizer.apply_gradients(zip(grads_t, [loss_object.trans_mat]))

        return logits, loss_value


@tf.function(experimental_relax_shapes=True)
def apply_gradient_markov(optimizer, loss_object, model, x, y):
    """
    Calculates and applies the gradient for a given batch and loss object
    for the DNN and HMM parameters given an optimizer

    Parameters
    ----------
    optimizer : tensorflow.keras.optimizers.Optimizer
       An optimizer for the DNN and HMM for Gradient Descent
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    model : tensorflow.python.keras.models.Model
        The DNN model
    x : tf.Tensor
       A tensor containing a batch of observation
    y : tf.Tensor
       A tensor containing the batch ground truth

    Returns
    -------
    tf.Tensor
        The value of the loss function given the method's parameters
    """
    with tf.GradientTape() as tape, tf.GradientTape() as p_tape, tf.GradientTape() as t_tape:
        tape.watch(x)
        p_tape.watch(loss_object.p_states)
        t_tape.watch(loss_object.trans_mat)
        logits = model(x, training=False)
        loss_value = loss_object(y_true=y, y_pred=logits)

        # HMM gradients
        grads_p = p_tape.gradient(loss_value, [loss_object.p_states])
        grads_t = t_tape.gradient(loss_value, [loss_object.trans_mat])

        optimizer.apply_gradients(zip(grads_p, [loss_object.p_states]))
        optimizer.apply_gradients(zip(grads_t, [loss_object.trans_mat]))

        return logits, loss_value


@tf.function(experimental_relax_shapes=True)
def apply_multi_gradient(optimizer_nn, optimizer_hmm, loss_object, model, x, y):
    """
    Calculates and applies the gradient for a given batch and loss object
    for the DNN and HMM parameters given two distinct optimizers

    Parameters
    ----------
    optimizer_nn : tensorflow.keras.optimizers.Optimizer
       An optimizer for the DNN for Gradient Descent
    optimizer_hmm : tensorflow.keras.optimizers.Optimizer
       An optimizer for the HMM for Gradient Descent
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    model : tensorflow.python.keras.models.Model
        The DNN model
    x : tf.Tensor
       A tensor containing a batch of observation
    y : tf.Tensor
       A tensor containing the batch ground truth

    Returns
    -------
    tf.Tensor
        The value of the loss function given the method's parameters
    """
    with tf.GradientTape() as tape, tf.GradientTape() as p_tape, tf.GradientTape() as t_tape:
        tape.watch(x)
        p_tape.watch(loss_object.p_states)
        t_tape.watch(loss_object.trans_mat)
        logits = model(x, training=True)
        loss_value = loss_object(y_true=y, y_pred=logits)

    # Neural Network gradient
    grads = tape.gradient(loss_value, model.trainable_variables)

    # HMM gradients
    grads_p = p_tape.gradient(loss_value, [loss_object.p_states])
    grads_t = t_tape.gradient(loss_value, [loss_object.trans_mat])

    # NN optimizer step
    optimizer_nn.apply_gradients(zip(grads, model.trainable_variables))
    # HMM optimizer step
    optimizer_hmm.apply_gradients(zip(grads_p, [loss_object.p_states]))
    optimizer_hmm.apply_gradients(zip(grads_t, [loss_object.trans_mat]))

    return logits, loss_value


def hmm_train_step_multi_opt(*, model, optimizer_nn, optimizer_hmm, train_batch, label_batch, loss_object, metrics):
    """
    Calculates the gradient according to some provided loss instance and applies it
    to both the HMM and DNN parameters using a projected gradient descent that guarantees
    that the models' parameters stay on the canonical simplex.
    This implementation uses different optimizers for the DNN and HMM

    Parameters
    ----------
    model : tensorflow.python.keras.models.Model
       The DNN model
    optimizer_nn : tensorflow.keras.optimizers.Optimizer
       An optimizer for the DNN for Gradient Descent
    optimizer_hmm : tensorflow.keras.optimizers.Optimizer
       An optimizer for the HMM for Gradient Descent
    train_batch : tf.Tensor
       A tensor containing a batch of observation
    label_batch : tf.Tensor
       A tensor containing the batch ground truth
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    metrics : list, optional
       A list of tensorflow.keras.metrics.Metric instances

    Returns
    -------
    tf.Tensor
       The value of the loss function given the method's parameters for the provided batch
    """
    # Apply gradient tape
    logits, loss_value = apply_multi_gradient(optimizer_nn, optimizer_hmm, loss_object, model, train_batch, label_batch)

    # Update stationary p_states and trans_matrix gradients after canonical simplex projection
    loss_object.p_states.assign(
        tf.Variable(simplex_projection(loss_object.p_states.numpy()), trainable=True, dtype=tf.float32))
    loss_object.trans_mat.assign(
        tf.Variable(simplex_projection(loss_object.trans_mat.numpy()), trainable=True, dtype=tf.float32))
    # Update train_metrics should they exist
    if metrics is not None and len(metrics) > 0:
        for metric in metrics:
            try:
                metric(logits, label_batch)
            except:
                metric(loss_value)

    return loss_value


def hmm_train_step_markov_only(*, model, optimizer, train_batch, label_batch, loss_object, metrics, window=False):
    """
       Calculates the gradient according to some provided loss instance and applies it
       to both the HMM and DNN parameters using a projected gradient descent that guarantees
       that the models' parameters stay on the canonical simplex

       Parameters
       ----------
       model : tensorflow.python.keras.models.Model
           The DNN model
       optimizer : tensorflow.keras.optimizers.Optimizer
           An optimizer for Gradient Descent
       train_batch : tf.Tensor
           A tensor containing a batch of observation
       label_batch : tf.Tensor
           A tensor containing the batch ground truth
       loss_object : tensorflow.keras.losses.Loss
           A Loss function instance
       metrics : list, optional
           A list of tensorflow.keras.metrics.Metric instances

       Returns
       -------
       tf.Tensor
           The value of the loss function given the method's parameters
       """

    # Apply gradient tape
    logits, loss_value = apply_gradient_markov(optimizer, loss_object, model, train_batch, label_batch)

    # Update stationary p_states and trans_matrix gradients after canonical simplex projection
    loss_object.trans_mat.assign(
        tf.Variable(project_matrix_row_components(loss_object.trans_mat.numpy()), trainable=True, dtype=tf.float32))
    loss_object.p_states.assign(
        tf.Variable(QR_steady_state_distribution(loss_object.trans_mat.numpy()), trainable=True, dtype=tf.float32))
    # Update train_metrics should they exist
    if metrics is not None and len(metrics) > 0:
        for metric in metrics:
            try:
                if window:
                    logits = get_averaged_predictions(label_batch, logits)
                metric.update_state(logits, label_batch)
            except:
                metric(loss_value)

    return loss_value


def hmm_train_step(*, model, optimizer, train_batch, label_batch, loss_object, metrics, window=False):
    """
    Calculates the gradient according to some provided loss instance and applies it
    to both the HMM and DNN parameters using a projected gradient descent that guarantees
    that the models' parameters stay on the canonical simplex

    Parameters
    ----------
    model : tensorflow.python.keras.models.Model
        The DNN model
    optimizer : tensorflow.keras.optimizers.Optimizer
        An optimizer for Gradient Descent
    train_batch : tf.Tensor
        A tensor containing a batch of observation
    label_batch : tf.Tensor
        A tensor containing the batch ground truth
    loss_object : tensorflow.keras.losses.Loss
        A Loss function instance
    metrics : list, optional
        A list of tensorflow.keras.metrics.Metric instances

    Returns
    -------
    tf.Tensor
        The value of the loss function given the method's parameters
    """

    # Apply gradient tape
    logits, loss_value = apply_gradient(optimizer, loss_object, model, train_batch, label_batch)

    # Update stationary p_states and trans_matrix gradients after canonical simplex projection
    loss_object.trans_mat.assign(
        tf.Variable(project_matrix_row_components(loss_object.trans_mat.numpy()), trainable=True, dtype=tf.float32))
    loss_object.p_states.assign(
        tf.Variable(QR_steady_state_distribution(loss_object.trans_mat.numpy()), trainable=True, dtype=tf.float32))
    # Update train_metrics should they exist
    if metrics is not None and len(metrics) > 0:
        for metric in metrics:
            try:
                if window:
                    logits = get_averaged_predictions(label_batch, logits)
                metric.update_state(logits, label_batch)
            except:
                metric(loss_value)

    return loss_value


def hmm_train_step_nn_only(*, model, optimizer, train_batch, label_batch, loss_object, metrics, window=False):
    """
    Calculates the gradient according to some provided loss instance and applies it
    only to the DNN parameters using a projected gradient descent

    Parameters
    ----------
    model : tensorflow.python.keras.models.Model
       The DNN model
    optimizer : tensorflow.keras.optimizers.Optimizer
       An optimizer for Gradient Descent
    train_batch : tf.Tensor
       A tensor containing a batch of observation
    label_batch : tf.Tensor
       A tensor containing the batch ground truth
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    metrics : list, optional
       A list of tensorflow.keras.metrics.Metric instances

    Returns
    -------
    tf.Tensor
       The value of the loss function given the method's parameters
    """
    # Apply gradient tape
    logits, loss_value = apply_gradient_nn_only(optimizer, loss_object, model, train_batch, label_batch)

    # Update train_metrics should they exist
    if metrics is not None and len(metrics) > 0:
        for metric in metrics:
            try:
                if window:
                    logits = get_averaged_predictions(label_batch, logits)
                metric.update_state(logits, label_batch)
            except:
                metric(loss_value)

    return loss_value


def test_step(*, model, loss_object, x_test, y_test, metrics):
    """
    Updates a set of metrics in inference for a given model

    Parameters
    ----------
    model : tensorflow.python.keras.models.Model
       The DNN model
    loss_object : tensorflow.keras.losses.Loss
       A Loss function instance
    x_test : tf.Tensor
       A tensor containing a batch of observations
    y_test : tf.Tensor
       A tensor containing the batch ground truth
    metrics : list, optional
       A list of tensorflow.keras.metrics.Metric instances

    Returns
    -------
    tf.Tensor
       The value of the loss function given the method's parameters
    """
    predictions = model(x_test, training=False)
    loss_value = loss_object(y_test, predictions)

    if metrics is not None and len(metrics) > 0:
        for metric in metrics:
            try:
                metric(predictions, y_test)
            except:
                metric(loss_value)


def train_HMM_parameters(labels_arr, one_hot=True):
    for labels in labels_arr:
        p_states = np.zeros((4,))
        trans_mat = np.zeros((4, 4))
        if one_hot:
            labels_cat = np.argmax(labels, axis=1)
        else:
            labels_cat = labels - 1

        for s in range(4):
            p_states[s] = len(np.argwhere(labels_cat == s)) / len(labels_cat)
            same_count = 0
            next_count = 0
            for j in range(len(labels_cat) - 1):
                if (labels_cat[j] == s) and (labels_cat[j + 1] == s):
                    same_count += 1
                if (labels_cat[j] == s) and (labels_cat[j + 1] == (s + 1) % 4):
                    next_count += 1
            if (same_count + next_count) == 0:
                trans_mat[s][s] = 0
                trans_mat[s][(s + 1) % 4] = 0
            else:
                trans_mat[s][s] = same_count / (same_count + next_count)
                trans_mat[s][(s + 1) % 4] = next_count / (same_count + next_count)

    return p_states, trans_mat


def hmm_single_observation(labels_cat):
    p_states = np.zeros((4,))
    trans_mat = np.zeros((4, 4))
    for s in range(4):
        p_states[s] = len(np.argwhere(labels_cat == s)) / len(labels_cat)
        same_count = 0
        next_count = 0
        for j in range(len(labels_cat) - 1):
            if (labels_cat[j] == s) and (labels_cat[j + 1] == s):
                same_count += 1
            if (labels_cat[j] == s) and (labels_cat[j + 1] == (s + 1) % 4):
                next_count += 1
        if (same_count + next_count) == 0:
            trans_mat[s][s] = 0
            trans_mat[s][(s + 1) % 4] = 0
        else:
            trans_mat[s][s] = same_count / (same_count + next_count)
            trans_mat[s][(s + 1) % 4] = next_count / (same_count + next_count)

    return p_states, trans_mat