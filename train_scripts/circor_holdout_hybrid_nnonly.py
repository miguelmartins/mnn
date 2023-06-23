import numpy as np
import tensorflow as tf

import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_processing.signal_extraction import CircorExtractor
from data_processing.data_transformation import HybridPCGDataPreparer, prepare_validation_data, get_train_test_indices
from custom_train_functions.hmm_train_step import train_HMM_parameters, hmm_train_step_nn_only
from loss_functions.MMI_losses import CompleteLikelihoodLoss
from models.custom_models import simple_convnet
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution


def main():
    patch_size = 64
    nch = 4
    num_epochs = 50
    number_folders = 1
    learning_rate = 1e-3
    split = .2

    # Read Circor #
    good_indices, features, labels, patient_ids = CircorExtractor.read_from_np(
        '../datasets/PCG/circor_final/springer_circor_dataset.npy',
        patch_size=patch_size)
    ######

    name = "circor_holdout_hybrid_nnonly"
    experiment_logger = PCGExperimentLogger(path='../results/hybrid/circor', name=name, number_folders=number_folders)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))
    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = simple_convnet(nch, patch_size)
    loss_object = CompleteLikelihoodLoss(tf.Variable(tf.zeros((4, 4)), trainable=True, dtype=tf.float32),
                                         tf.Variable(tf.zeros((4,)), trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
    model.save_weights('random_init')  # Save initialization before training

    acc_folds, prec_folds = [], []

    indices = np.arange(len(features))
    idx_train, idx_test = train_test_split(indices, test_size=split, random_state=42)
    idx_train_ = []
    for idx in range(len(idx_train)):
        if not (patient_ids[idx_train[idx]] in patient_ids[idx_test]):
            idx_train_.append(idx_train[idx])
    idx_train = np.array(idx_train_)

    X_train, y_train = features[idx_train], labels[idx_train]
    X_dev, y_dev = features[idx_test], labels[idx_test]
    min_val_loss = 1e100
    model.load_weights('random_init')  # Load random weights f.e. fold

    print('Number of training sounds:', len(idx_train))
    print('Number of testing sounds:', len(idx_test))

    dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
    dp.set_features_and_labels(X_train, y_train)
    train_dataset = tf.data.Dataset.from_generator(dp,
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(None, patch_size, nch),
                                                                     dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                   ).prefetch(buffer_size=tf.data.AUTOTUNE)
    dev_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
    dev_dp.set_features_and_labels(X_dev, y_dev)
    dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, patch_size, nch), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                 ).prefetch(buffer_size=tf.data.AUTOTUNE)


    # MLE Estimation for HMM
    dataset_np = list(train_dataset.as_numpy_iterator())
    dataset = np.array(dataset_np, dtype=object)
    labels_ = dataset[:, 1]
    _, trans_mat = train_HMM_parameters(labels_)
    p_states = QR_steady_state_distribution(trans_mat)
    loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
    loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

    train_dataset = train_dataset.shuffle(len(X_train), reshuffle_each_iteration=True)

    best_p_states = loss_object.p_states.numpy()
    best_trans_mat = loss_object.trans_mat.numpy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)]
    print(loss_object.trans_mat.numpy())
    for ep in range(num_epochs):
        print('=', end='')
        metrics[0].reset_states()
        for i, (x_train, y_train) in tqdm(enumerate(train_dataset), desc=f'training', total=len(X_train),
                                          leave=True):
            hmm_train_step_nn_only(model=model,
                                   optimizer=optimizer_nn,
                                   loss_object=loss_object,
                                   train_batch=x_train,
                                   label_batch=y_train,
                                   metrics=metrics)

        print(f"ep: {ep}. Cat accuracy: {metrics[0].result().numpy()}")
        print(loss_object.trans_mat.numpy())
        print(loss_object.p_states.numpy())
        # check performance at each epoch
        (train_loss, train_acc) = model.evaluate(train_dataset, verbose=0)
        print("Train Loss", train_loss, "Train_acc", train_acc)
        (val_loss, val_acc) = model.evaluate(dev_dataset, verbose=0)
        print("Epoch ", ep, " train loss = ", train_loss, " train accuracy = ", train_acc, "val loss = ", val_loss,
              "val accuracy = ", val_acc)
        checkpoint = './checkpoints/hybrid_physion/' + '0' + '/my_checkpoint'
        print(loss_object.trans_mat.numpy())
        print(loss_object.p_states.numpy())
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            # SAVE MODEL
            best_p_states = loss_object.p_states.numpy()
            best_trans_mat = loss_object.trans_mat.numpy()
            experiment_logger.save_model_checkpoints(model, best_p_states, best_trans_mat, '/cnn_weights_fold_',
                                                     0)

    model = experiment_logger.load_model_checkpoint_weights(model)
    loss_object.p_states.assign(tf.Variable(best_p_states, trainable=True, dtype=tf.float32))
    loss_object.trans_mat.assign(tf.Variable(best_trans_mat, trainable=True, dtype=tf.float32))

    # collecting data and results
    experiment_logger.update_results(fold=0,
                                     train_indices=idx_train,
                                     test_indices=idx_test,
                                     output_seqs=np.array([], dtype=object),
                                     predictions=np.array([], dtype=object),
                                     ground_truth=np.array([], dtype=object))

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())


if __name__ == '__main__':
    main()
