
import numpy as np
import tensorflow as tf
import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint

from data_processing.signal_extraction import DataExtractor, CircorExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
    prepare_validation_data, get_train_test_indices
from custom_train_functions.hmm_train_step import train_HMM_parameters
from models.custom_models import bilstm_attention_fernando19_softmax
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution
from tqdm import tqdm
import os

def scheduler(epoch, lr): return lr * 0.1 if epoch == 10 else lr


def main():
    patch_size = 64
    n_mfcc = 6
    nch = n_mfcc * 3
    num_epochs = 50
    number_folders = 10
    learning_rate = initial_learning_rate = 0.002
    batch_size = 32
    _, patient_ids, features, labels = CircorExtractor.from_mat('../datasets/circor_final_labels50hz.mat'
                                                             , patch_size=patch_size)
    features = CircorExtractor.normalize_signal(features)
    features = DataExtractor.get_mfccs(data=features,
                                       sampling_rate=1000,
                                       window_length=150,
                                       window_overlap=130,
                                       n_mfcc=6)
    # features = CircorExtractor.align_psd_labels(features, labels)
    good_indices = CircorExtractor.filter_smaller_than_patch(patch_size, features)
    name = 'fernando_CE_physio16_mfcc_joint'
    experiment_logger = PCGExperimentLogger(path='../results/fernando/circor', name=name, number_folders=number_folders)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))
    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = bilstm_attention_fernando19_softmax(nch, patch_size)
    model.save_weights('random_init_lstm_attention')  # Save initialization before training

    acc_folds, prec_folds = [], []
    for j_fold in range(number_folders):
        min_val_loss = 1e3
        optimizer_nn = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        model.compile(optimizer=optimizer_nn, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.load_weights('random_init_lstm_attention')  # Load random weights f.e. fold
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices,
                                                             number_folders=number_folders,
                                                             patient_ids=patient_ids,
                                                             fold=j_fold)

        # remove from training data sounds that are from patient appearing in the testing set

        print('Considering folder number:', j_fold + 1)
        # This ia residual code from the initial implementation, kept for "panicky" reasons
        features_train = features[train_indices]
        features_test = features[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]

        train_indices = good_indices[train_indices]
        test_indices = good_indices[test_indices]

        print('Number of training sounds:', len(labels_train))
        print('Number of testing sounds:', len(labels_test))

        X_train, X_dev, y_train, y_dev = train_test_split(
            features_train, labels_train, test_size=0.1, random_state=42)
        _, trans_mat = train_HMM_parameters(y_train, one_hot=False)
        p_states = QR_steady_state_distribution(trans_mat)

        dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        dp.set_features_and_labels(X_train, y_train)
        train_dataset = tf.data.Dataset.from_generator(dp,
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                         dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                       )
        dev_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        dev_dp.set_features_and_labels(X_dev, y_dev)
        dev_dataset = tf.data.Dataset.from_generator(dev_dp,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                     )

        test_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      )

        train_dataset = train_dataset.shuffle(buffer_size=400, reshuffle_each_iteration=True)
        checkpoint_path = experiment_logger.path + '/weights_fold' + str(j_fold) + '.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
        schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
        history = model.fit(train_dataset, validation_data=dev_dataset,
                            validation_steps=1, batch_size=batch_size,
                            epochs=num_epochs, verbose=1,
                            shuffle=True, callbacks=[model_checkpoint, schedule])
        experiment_logger.save_markov_state(j_fold, p_states, trans_mat)
        model.load_weights(checkpoint_path)
        out_test = model.predict(test_dataset)
        accuracy, precision = [], []

        labels_list, predictions_list = [], []

        # Viterbi algorithm in test set
        for x, y in tqdm(test_dataset, desc=f'validating (viterbi)', total=len(labels_test)):
            logits = model.predict(x)
            y = y.numpy()
            _, _, predictions = log_viterbi_no_marginal(p_states, trans_mat,
                                                        logits)
            predictions = predictions.astype(np.int32)
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            predictions_list.append(predictions)
            labels_list.append(raw_labels)
            acc = accuracy_score(raw_labels, predictions)
            accuracy.append(acc)
            # precision.append(prc)
        acc_folds.append(np.mean(acc))
        prec_folds.append(np.mean(acc))
        length_sounds_test = np.zeros(len(features_test))
        for j in range(len(features_test)):
            length_sounds_test[j] = len(features_test[j])

        # recover sound labels from patch labels
        output_probs, output_seqs = prepare_validation_data(out_test, test_indices, length_sounds_test)

        sample_acc = np.zeros((len(labels_test),))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (np.sum((output_seqs[j] != labels_test[j] - 1)) / len(labels_test[j]))

        print('Test mean sample accuracy for this folder:', np.sum(sample_acc) / len(sample_acc))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (
                    np.sum((predictions_list[j] != labels_test[j] - 1).astype(int)) / len(labels_test[j]))
        print("Viterbi: ", np.sum(sample_acc) / len(sample_acc))

        # collecting data and results
        experiment_logger.update_results(fold=j_fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=output_seqs,
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=p_states,
                                   trans_mat=trans_mat)


if __name__ == '__main__':
    main()
