import os.path

import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data_processing.signal_extraction import DataExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
    prepare_validation_data, get_train_test_indices
from custom_train_functions.hmm_train_step import train_HMM_parameters
from models.custom_models import bilstm_attention_fernando19_softmax
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution
from tqdm import tqdm


def scheduler(epoch, lr): return lr * 0.1 if epoch == 10 else lr


def main():
    patch_size = 64
    n_mfcc = 6
    nch = n_mfcc * 3
    num_epochs = 50
    number_folders = 1
    learning_rate = 0.002
    batch_size = 32
    split = .1

    good_indices, _, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                     '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                patch_size=patch_size)
    features_, _ = DataExtractor.read_physionet_mat(
        '../datasets/PCG/example_data.mat')  # TODO: o som original deve ter 20x mais samples
    length_sounds = np.array([len(features_[j]) for j in range(len(features_))])
    features = DataExtractor.filter_by_index(features_, good_indices)
    features = DataExtractor.get_mfccs(data=features,
                                       sampling_rate=1000,
                                       window_length=150,
                                       window_overlap=130,
                                       n_mfcc=6)
    name = 'fernando_CE_physio16_mfcc_joint'
    experiment_logger = PCGExperimentLogger(path='../results/rerun/fernando/ph_holdout', name=name, number_folders=number_folders)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))
    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = bilstm_attention_fernando19_softmax(nch, patch_size)
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_nn, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.save_weights('random_init_lstm_attention')  # Save initialization before training

    acc_folds, prec_folds = [], []
    min_val_loss = 1e3
    model.load_weights('random_init_lstm_attention')  # Load random weights f.e. fold
    indices = np.arange(len(features))
    idx_train, idx_test = get_train_test_indices(good_indices=good_indices,
                                                 number_folders=10,
                                                 patient_ids=patient_ids,
                                                 fold=0)

    X_train, y_train = features[idx_train], labels[idx_train]
    X_dev, y_dev = features[idx_test], labels[idx_test]
    idx_train, idx_test = good_indices[idx_train], good_indices[idx_test]

    # NORMALIZAR PSD
    # como separar as features para a nossa CNN?
    # com os envolopes separámos em patches, aqui usamos a própria dimensão da STFT?
    # Contruir datapreparer:
    #    - Separe o som por janelas PSD
    #
    # Implementar uma CNN com convoluções 2D: first approach -> mudar a nossa CNN para operações 2D.
    # ???
    # Profit
    dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
    dp.set_features_and_labels(X_train, y_train)
    train_dataset = tf.data.Dataset.from_generator(dp,
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                     dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                   ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    dev_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
    dev_dp.set_features_and_labels(X_dev, y_dev)
    dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                   dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                 ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    # MLE Estimation for HMM
    dataset_np = list(train_dataset.as_numpy_iterator())
    dataset = np.array(dataset_np, dtype=object)
    labels_ = dataset[:, 1]
    _, trans_mat = train_HMM_parameters(labels_)
    p_states = QR_steady_state_distribution(trans_mat)

    train_dataset = train_dataset.shuffle(buffer_size=400, reshuffle_each_iteration=True)
    checkpoint_path = experiment_logger.path + '/weights_fold.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    history = model.fit(train_dataset, validation_data=dev_dataset,
                        validation_steps=1, batch_size=batch_size,
                        epochs=num_epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint, schedule])
    experiment_logger.save_markov_state(0, p_states, trans_mat)
    model.load_weights(checkpoint_path)
    out_test = model.predict(dev_dataset)
    accuracy, precision = [], []

    labels_list, predictions_list = [], []

    # Viterbi algorithm in test set
    for x, y in tqdm(dev_dataset, desc=f'validating (viterbi)', total=len(y_dev), leave=True):
        logits = model.predict(x)
        y = y.numpy()
        _, _, predictions = log_viterbi_no_marginal(p_states, trans_mat,
                                                    logits)
        predictions = predictions.astype(np.int32)
        raw_labels = np.argmax(y, axis=1).astype(np.int32)
        predictions_list.append(predictions)
        labels_list.append(raw_labels)
        acc = accuracy_score(raw_labels, predictions)
        prc = precision_score(raw_labels, predictions, average=None)
        accuracy.append(acc)
        precision.append(prc)
    print("Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
    acc_folds.append(np.mean(acc))
    prec_folds.append(np.mean(prc))
    length_sounds_test = np.zeros(len(X_dev))
    for j in range(len(X_dev)):
        length_sounds_test[j] = len(X_dev[j])

    # recover sound labels from patch labels
    output_probs, output_seqs = prepare_validation_data(out_test, X_dev, length_sounds_test)

    sample_acc = np.zeros((len(X_dev),))
    for j in range(len(X_dev)):
        sample_acc[j] = 1 - (np.sum((output_seqs[j] != X_dev[j] - 1)) / len(X_dev[j]))

    print('Test mean sample accuracy for this folder:', np.sum(sample_acc) / len(sample_acc))
    for j in range(len(y_dev)):
        sample_acc[j] = 1 - (
                np.sum((predictions_list[j] != y_dev[j] - 1).astype(int)) / len(y_dev[j]))

    # collecting data and results
    experiment_logger.update_results(fold=0,
                                     train_indices=good_indices,
                                     test_indices=idx_test,
                                     output_seqs=output_seqs,
                                     predictions=np.array(predictions_list, dtype=object),
                                     ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=p_states,
                                   trans_mat=trans_mat)


if __name__ == '__main__':
    main()
