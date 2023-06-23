import os

import h5py
import numpy as np
import tensorflow as tf
import sys
from tempfile import mkdtemp

import os.path as path
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_processing.signal_extraction import DataExtractor, CircorExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
    prepare_validation_data, get_train_test_indices, HybridPCGDataPreparer
from tqdm import tqdm
from custom_train_functions.hmm_train_step import hmm_train_step, train_HMM_parameters, hmm_train_step_markov_only, \
    hmm_single_observation
from loss_functions.MMI_losses import CompleteLikelihoodLoss, ForwardLoss
from models.custom_models import simple_convnet2d, simple_convnet
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal, QR_steady_state_distribution
BASE_PATH = '../results/hybrid/circor/circor_holdout_hybrid_nnonly/2023-06-06_11:24:00'
MODEL_CKTPT = os.path.join(BASE_PATH, 'cnn_weights_fold_0')
TRANS_MAT_CKPT = os.path.join(BASE_PATH, 'trans_mat_fold_0.npy')
P_STATES_CKPT = os.path.join(BASE_PATH, 'p_states_fold_0.npy')


def main():
    patch_size = 64
    nch = 4
    num_epochs = 50
    number_folders = 10
    learning_rate = 1e-3

    # Read Ph16 #
    good_indices, features, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                            '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                       patch_size=patch_size)
    ######

    name = "finetune_1_50_hmm_only_circor_to_ph16"
    experiment_logger = PCGExperimentLogger(path='../results/rerun/hybrid/ph16', name=name,
                                            number_folders=number_folders)

    h5_dir = os.path.join(experiment_logger.path, 'results.hdf5')
    h5_file = h5py.File(h5_dir, 'w')
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))

    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints

    trans_mat = np.load(TRANS_MAT_CKPT)
    p_states = np.load(P_STATES_CKPT)
    model = simple_convnet(nch, patch_size)
    loss_object = ForwardLoss(tf.Variable(trans_mat, trainable=True, dtype=tf.float32),
                              tf.Variable(p_states, trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def load_configs():
        model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
        model.load_weights(MODEL_CKTPT)  # Save initialization before training

        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))

    acc_folds, prec_folds = [], []
    for j_fold in range(number_folders):
        min_val_loss = 1e100
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

        test_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      )
        accuracy, precision = [], []

        labels_list, predictions_list = [], []

        # Viterbi algorithm in test set
        h5_file[f'/fold{j_fold}/data'] = f'{j_fold}'
        for i, (x, y) in tqdm(enumerate(test_dataset), desc=f'validating (viterbi)', total=len(labels_test),
                              leave=True):
            load_configs()
            y = y.numpy()
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            h5_file[f'/fold{j_fold}/data'].attrs[f'gt_{i}'] = raw_labels
            pred_tensor = []
            for it in range(num_epochs):
                loss_value = hmm_train_step(model=model,
                                            optimizer=optimizer_nn,
                                            loss_object=loss_object,
                                            train_batch=x,
                                            label_batch=y,
                                            metrics=None)

                logits = model.predict(x)
                _, _, predictions = log_viterbi_no_marginal(loss_object.p_states.numpy(), loss_object.trans_mat.numpy(),
                                                            logits)
                h5_file[f'/fold{j_fold}/data'].attrs[f'preds{i}_{it}'] = predictions.astype(np.int32)

                # print(f'Target')
                #print("*****")


            predictions_list.append(predictions.astype(np.int32))
            labels_list.append(raw_labels)
        print("Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
        length_sounds_test = np.zeros(len(features_test))
        for j in range(len(features_test)):
            length_sounds_test[j] = len(features_test[j])

        # recover sound labels from patch labels
        out_test = model.predict(test_dataset)
        output_probs, output_seqs = prepare_validation_data(out_test, test_indices, length_sounds_test)

        sample_acc = np.zeros((len(labels_test),))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (np.sum((output_seqs[j] != labels_test[j] - 1).astype(int)) / len(labels_test[j]))

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

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())
    # chec
    h5_file.close()


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
