import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

from data_processing.signal_extraction import DataExtractor
from data_processing.data_transformation import HybridPCGDataPreparer2D, \
    prepare_validation_data, get_train_test_indices
from tqdm import tqdm
from custom_train_functions.hmm_train_step import hmm_train_step, train_HMM_parameters
from loss_functions.MMI_losses import CompleteLikelihoodLoss, ForwardLoss
from models.custom_models import simple_convnet2d
from utility_functions.experiment_logs import PCGExperimentLogger

from utility_functions.hmm_utilities import log_viterbi_no_marginal


def main():
    patch_size = 64
    nch = 76
    num_epochs = 10
    number_folders = 10
    learning_rate = 1e-3
    SAVED_PATH = '../results/hybrid/hmm_completlikelihood1e3_physio16_psd_joint/2022-03-21_15:58:21'
    good_indices, _, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                     '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                patch_size=patch_size)
    features_, _ = DataExtractor.read_physionet_mat(
        '../datasets/PCG/example_data.mat')  # TODO: o som original deve ter 20x mais samples
    length_sounds = np.array([len(features_[j]) for j in range(len(features_))])
    features = DataExtractor.filter_by_index(features_, good_indices)
    features = DataExtractor.get_power_spectrum(data=features,
                                                sampling_rate=1000,
                                                window_length=150,
                                                window_overlap=130,
                                                window_type='hann')
    name = 'hmm_completlikelihood1e3_physio16_psd_joint'
    experiment_logger = PCGExperimentLogger(path='../results/hybrid/fine_tune', name=name,
                                            number_folders=number_folders)
    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))
    # 1) save files on a given directory, maybe experiment-name/date/results
    # 2) save model weights (including random init, maybe  experiment-name/date/checkpoints
    model = simple_convnet2d(nch, patch_size)
    loss_object = ForwardLoss(tf.Variable(tf.zeros((4, 4)), trainable=True, dtype=tf.float32),
                              tf.Variable(tf.zeros((4,)), trainable=True, dtype=tf.float32))
    optimizer_nn = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])
    model.save_weights('random_init')  # Save initialization before training

    acc_folds, prec_folds = [], []
    for j_fold in range(number_folders):
        min_val_loss = 1e3
        model.load_weights('random_init')  # Load random weights f.e. fold
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices,
                                                             number_folders=number_folders,
                                                             patient_ids=patient_ids,
                                                             fold=j_fold)

        # remove from training data sounds that are from patient appearing in the testing set

        print('Considering folder number:', j_fold + 1)

        features_train = features[train_indices]
        features_test = features[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]

        # This ia residual code from the initial implementation, kept for "panicky" reasons
        train_indices = good_indices[train_indices]
        test_indices = good_indices[test_indices]

        print('Number of training sounds:', len(labels_train))
        print('Number of testing sounds:', len(labels_test))
        test_dp = HybridPCGDataPreparer2D(patch_size=patch_size, number_channels=nch, num_states=4)
        test_dp.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(test_dp,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch, 1),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        # MLE Estimation for HMM
        model, p_states, trans_mat = experiment_logger.stored_model_checkpoints(model=model,
                                                                                path=SAVED_PATH,
                                                                                cnn_prefix='cnn_weights_fold_',
                                                                                fold=j_fold)
        model.trainable = True
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))
        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        # model.compile(optimizer=optimizer_nn, loss=loss_object, metrics=['categorical_accuracy'])

        out_test = model.predict(test_dataset)
        accuracy, precision = [], []

        labels_list, cnn_list, predictions_list = [], [], []
        print(loss_object.p_states.numpy())
        print(loss_object.trans_mat.numpy())
        acc_cnn = []
        # Viterbi algorithm in test set
        for epoch in range(num_epochs):
            # model.trainable = True
            for x, y in tqdm(test_dataset, desc=f'fine_tuning p(O|model)', total=len(labels_test), leave=True):
                hmm_train_step(model=model,
                               optimizer=optimizer_nn,
                               loss_object=loss_object,
                               train_batch=x,
                               label_batch=y,
                               metrics=None)

        for x, y in tqdm(test_dataset, desc=f'VALIDATING p(O|model)', total=len(labels_test), leave=True):
            logits = model.predict(x)
            vit, _, predictions = log_viterbi_no_marginal(loss_object.p_states.numpy(),
                                                          loss_object.trans_mat.numpy(),
                                                          logits)
            predictions = predictions.astype(np.int32)
            cnn_preds = np.argmax(logits, axis=1)
            cnn_list.append(cnn_preds)
            raw_labels = np.argmax(y, axis=1).astype(np.int32)
            predictions_list.append(predictions)
            labels_list.append(raw_labels)
            acc = accuracy_score(raw_labels, predictions)
            acc_cnn.append(accuracy_score(raw_labels, cnn_preds))
            prc = precision_score(raw_labels, predictions, average=None)
            accuracy.append(acc)
            precision.append(prc)

        print("CNN-> Average Accuracy", np.mean(acc_cnn))
        print("Viterbi-> Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
        acc_folds.append(np.mean(acc))
        prec_folds.append(np.mean(prc))

        # collecting data and results
        experiment_logger.update_results(fold=j_fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=np.array(cnn_list, dtype=object),
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=loss_object.p_states.numpy(),
                                   trans_mat=loss_object.trans_mat.numpy())


if __name__ == '__main__':
    main()
